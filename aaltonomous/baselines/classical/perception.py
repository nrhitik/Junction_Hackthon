"""Head RGB-D -> coloured objects in the pelvis frame. Shape-agnostic: an object is a coloured blob, its grasp
target is the centre of its top face.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw), ...]

Per colour: HSV threshold (config.yaml perception.hsv) -> opening -> connected components. Per blob: erode, take
the pixels with valid depth, deproject them (calibration.yaml intrinsics + T_pelvis_optical), keep the top face
(points within `top_band` of a robust max height) and use the median of those as the top centre. With too few valid
depth pixels (holes, or no depth stream) the blob centroid ray is intersected with the plane at the known object
height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong height are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "plane"
    yaw: float | None = None   # in-plane orientation of the top face (rad), info only
    uv: tuple = field(default=(0.0, 0.0))
    # team: shape from the top-face outline (pelvis xy): "cube" | "cylinder" | "triangle" | None (no depth / too few
    # points). grasp_dirs: unit pelvis-xy finger directions that pinch across flats (thumb on the -dir side):
    # cube = its 4 face normals; triangle = flat -> opposite apex (3). centroid_to_pinch: m along each dir from the
    # top-face centroid to the middle of the pinch (triangle: the centroid is 1/3 altitude from the flat).
    shape: str | None = None
    grasp_dirs: list | None = None
    centroid_to_pinch: float = 0.0
    shape_fit: dict | None = None
    top_xy: object = None          # team: top-face points (pelvis xy) for a later forced-shape orientation
    shape_src: str | None = None   # team: "ours" | "onnx" | "both" (who decided the shape)

    def __repr__(self):
        x, y, z = self.top
        return (f"{self.color:<6} {self.shape or '?':8s} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} "
                f"depth_px={self.n_depth} [{self.method}]")


class Perception:
    def __init__(self, cfg, scene, calibration):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # pelvis-frame xy correction added to every detection (head extrinsics error; tuned on the robot)
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        out = {}
        for color, ranges in self.cfg["hsv"].items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def color_points(self, rgb, depth, color, step=2):
        """Team feedback layer: pelvis-frame 3D points (n, 3) of every `color` pixel with valid depth (every `step`-th
        pixel). No bin / height / mat filtering: the caller decides where it looks (hand, original spot, bin)."""
        if depth is None:
            return np.zeros((0, 3))
        m = self.masks(rgb)[color]
        if step > 1:
            sub = np.zeros_like(m)
            sub[::step, ::step] = m[::step, ::step]
            m = sub
        v, u = np.nonzero(m)
        d = depth[v, u].astype(float)
        dlo, dhi = self.cfg["depth_range_mm"]
        ok = (d >= dlo) & (d <= dhi)
        if not ok.any():
            return np.zeros((0, 3))
        P = self.cal.pixel_to_pelvis(self.cam, u[ok], v[ok], d[ok])
        P[:, :2] += self.xy_offset
        return P

    def detect(self, rgb, depth, debug=False):
        import cv2
        c = self.cfg
        dets, rejected = [], []
        dlo, dhi = c["depth_range_mm"]
        keep = self._keep_mask(depth)
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < c["min_area_px"]:
                    continue
                blob = (lab == i).astype(np.uint8)
                e = int(c["erode_px"])
                core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                if core.sum() < 0.25 * area:
                    core = blob
                v, u = np.nonzero(core)
                d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
                ok = (d >= dlo) & (d <= dhi)
                det = None
                if ok.sum() >= c["min_valid_depth_px"]:
                    det = self._from_depth(color, u[ok], v[ok], d[ok], area)
                if det is None:
                    det = self._from_plane(color, cent[i], area, int(ok.sum()))
                why = self._reject(det)
                if why:
                    rejected.append((det, why))
                else:
                    dets.append(det)
        dets, merged = merge_same_object(dets, float(c.get("merge_radius_m", 0.04)))
        rejected += merged
        return (dets, rejected) if debug else dets

    def measure_box_front(self, depth, step=3, exclude_xy=(), exclude_r=0.045):
        """team: pelvis x of the bin row's FRONT edge, measured from head depth (the real boxes stood ~10 cm further
        than the model). Per bin: points 2-15 cm above the counter in a +-7 cm band around the bin's y; per 1 cm
        y-slice the nearest x; the 75th percentile over slices (objects in front cover only some slices).
        Returns (front_x or None, {bin: front_x})."""
        sc = self.scene
        if depth is None:
            return None, {}
        d = np.asarray(depth, float)[::step, ::step]
        vv, uu = np.nonzero((d >= 150) & (d <= 2000))
        if len(vv) < 500:
            return None, {}
        P = self.cal.pixel_to_pelvis(self.cam, uu * step, vv * step, d[vv, uu])
        h = P[:, 2] - sc.counter_z
        sel = (h > 0.02) & (h < 0.15) & (P[:, 0] > 0.36) & (P[:, 0] < 0.85)
        for xy in exclude_xy:   # points of detected objects (standing in front of the boxes) do not count
            sel &= np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) > exclude_r
        P = P[sel]
        fronts = {}
        for name, ctr in sc.cfg["bins"]["centers"].items():
            cy = sc.mat_to_pelvis(np.r_[ctr, 0.0])[1]
            band = P[np.abs(P[:, 1] - cy) < 0.07]
            if len(band) < 50:
                continue
            mins = []
            for y0 in np.arange(cy - 0.07, cy + 0.07, 0.01):
                sl = band[(band[:, 1] >= y0) & (band[:, 1] < y0 + 0.01), 0]
                if len(sl) >= 3:
                    mins.append(float(np.min(sl)))
            if len(mins) >= 8:
                fronts[name] = float(np.percentile(mins, 75))
        # the CENTER box is straight ahead and never hidden by a hand: it decides; side boxes are often partly
        # hidden by the hands at home (their back wall gets measured) -> used only when the center is missing
        if "center" in fronts:
            return fronts["center"], fronts
        v = np.array(list(fronts.values()))
        if len(v) < 2 or v.max() - v.min() > 0.04:
            return None, fronts
        return float(np.median(v)), fronts

    def _keep_mask(self, depth):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        keep = self._bin_hulls_mask(depth.shape if depth is not None else (480, 640))
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        ok = (h >= 0.25) & (h <= c["height_range"][1])
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _from_depth(self, color, u, v, d, area):
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        xy = xy + self.xy_offset
        det = Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                        (float(np.median(u)), float(np.median(v))))
        if self.cfg.get("shape", {}).get("enabled", False):
            det.top_xy = top[:, :2] + self.xy_offset   # kept: orientation for a shape decided later (ONNX fusion)
            _classify_shape(det, top[:, :2] + self.xy_offset, self.size, self.cfg["shape"])
        return det

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.cal._T(self.cam)
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det):
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        return None


def merge_same_object(dets, radius=0.04):
    """team: ONE physical object = ONE detection. Colour masks are grouped per colour, so a shaded yellow face can
    become a second, 'orange' blob on the same cube (real robot, 2026-11-15). Detections whose 3D top centres are
    closer than `radius` (layouts keep objects >= 8 cm apart) are one object: keep the one with the HIGHEST top
    (the lit top face carries the true colour), ties -> the bigger blob. Returns (kept, [(dropped, reason)])."""
    order = sorted(dets, key=lambda d: (-float(d.top[2]), -int(d.n_px)))
    kept, dropped = [], []
    for d in order:
        near = [k for k in kept if np.linalg.norm(np.asarray(k.top[:2]) - np.asarray(d.top[:2])) < radius]
        if near:
            dropped.append((d, f"same object as {near[0].color} at ({near[0].top[0]:+.2f},{near[0].top[1]:+.2f})"))
        else:
            kept.append(d)
    return kept, dropped


def _classify_shape(det, xy, size, cfg, force=None):
    """Team: classify the top face from its pelvis-xy points and set det.shape / grasp_dirs / centroid_to_pinch.

    Fill ratios of the convex hull: vs its min-area rectangle (square = 1, circle = 0.79, triangle = 0.5), vs its
    min enclosing circle (circle = 1, square = 0.64, triangle = 0.41) and vs its min enclosing triangle (triangle = 1,
    circle = 0.60, square = 0.5). The largest wins. The hull is of the top-band points, so the outline is shrunk by the
    erosion; fills are scale-free, so that does not matter. A hull much smaller than the object = unreliable -> None.
    """
    import cv2
    if len(xy) < int(cfg.get("min_points", 40)):
        return
    pts = (np.asarray(xy, float) * 1000.0).astype(np.float32)  # mm: OpenCV geometry wants a reasonable scale
    hull = cv2.convexHull(pts)
    a_h = float(cv2.contourArea(hull))
    if a_h < float(cfg.get("min_area_frac", 0.3)) * (size * 1000.0) ** 2:
        return
    (_, _), (w, h), ang = cv2.minAreaRect(hull)
    (_, _), r = cv2.minEnclosingCircle(hull)
    a_t, tri = cv2.minEnclosingTriangle(hull)
    fit = {"cube": a_h / max(w * h, 1e-6), "cylinder": a_h / max(np.pi * r * r, 1e-6),
           "triangle": a_h / max(float(a_t), 1e-6)}
    if force is None:
        det.shape_fit = {k: round(float(v), 3) for k, v in fit.items()}
    shape = force or max(fit, key=fit.get)
    if force is None and fit[shape] < float(cfg.get("min_fit", 0.75)):
        return
    if shape == "triangle" and force is None:
        # a triangle grasp on a non-triangle shifts the grasp point: demand a clear margin, else no shape (generic)
        runner_up = max(v for k, v in fit.items() if k != "triangle")
        if fit["triangle"] - runner_up < float(cfg.get("triangle_margin", 0.1)):
            return
    det.shape = shape
    if shape == "cube":
        a = np.radians(ang)
        det.grasp_dirs = [np.array([np.cos(a + k * np.pi / 2), np.sin(a + k * np.pi / 2)]) for k in range(4)]
    elif shape == "triangle":
        T = tri.reshape(3, 2).astype(float)
        dirs = []
        for i in range(3):  # thumb on the flat opposite vertex i, index on vertex i
            mid = 0.5 * (T[(i + 1) % 3] + T[(i + 2) % 3])
            d = T[i] - mid
            dirs.append(d / max(np.linalg.norm(d), 1e-9))
        det.grasp_dirs = dirs
        # top-face centroid is 1/3 of the altitude from the flat; the pinch middle is at 1/2 -> +altitude/6
        det.centroid_to_pinch = float(cfg.get("triangle_pinch_shift", size / 6.0))


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def fuse_onnx_shapes(dets, rgb, size, cfg):
    """Team: combine our top-face shape with the teammate's ONNX classifier (aaltonomous.brain.shapes) on the head
    image. Sim test (24 objects): ONNX prisms 7/7, cylinders 6/6, cubes 6/11 (4 called prism); ours cubes 8/10.
      ONNX cylinder -> cylinder | ONNX cube -> cube | ONNX prism -> cube if ours says cube (or our fits favour cube),
      else prism | ONNX nothing -> ours. The orientation (grasp_dirs) always comes from our 3D top-face points."""
    import cv2
    from aaltonomous.brain.colors import find_objects
    from aaltonomous.brain.shapes import classify
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    objs = find_objects(bgr)
    if not objs:
        return dets
    res = classify(bgr, objs)
    name = {"prism": "triangle", "cube": "cube", "cylinder": "cylinder"}
    for d in dets:
        u, v = d.uv
        k = min(range(len(objs)), key=lambda i: (objs[i].cx - u) ** 2 + (objs[i].cy - v) ** 2)
        if np.hypot(objs[k].cx - u, objs[k].cy - v) > float(cfg.get("onnx_match_px", 35)):
            d.shape_src = "ours" if d.shape else None
            continue
        o, p = name.get(res[k][0]), res[k][1]
        ours = d.shape
        if o == "triangle" and (ours == "cube" or (ours is None and d.shape_fit and
                                                   max(d.shape_fit, key=d.shape_fit.get) == "cube")):
            final = "cube"
        else:
            final = o
        d.shape_src = "both" if final == ours else "onnx"
        if final != ours:
            d.shape, d.grasp_dirs, d.centroid_to_pinch = None, None, 0.0
            if d.top_xy is not None:
                _classify_shape(d, d.top_xy, size, cfg.get("shape", {}), force=final)
            d.shape = final
    return dets


def locate_lid_holes(rgb, depth, cal, cam, expected, tol=0.029):
    """Team: tune the lid hole positions with the head camera. The teammate's find_lid (aaltonomous.brain.lid) finds the
    lid and the holes it can see (image u, v); head DEPTH gives the lid plate's real height (median of plate pixels in a
    ring around each hole); the hole centre = camera ray through (u, v) meets that plane. Each hole may move at most
    `tol` (m, per axis) from `expected` ({shape: {"xy", "z", ...}}, pelvis frame); holes not seen keep their expected
    position. Returns (new holes dict, report list)."""
    import copy
    import cv2
    from aaltonomous.brain.lid import find_lid
    out, report = copy.deepcopy(expected), []
    if rgb is None or depth is None:
        return out, ["lid: no image/depth"]
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    lid = find_lid(bgr, None)
    if lid is None:
        return out, ["lid: not found by the camera, using the expected holes"]
    name = {"cube": "cube", "cylinder": "cylinder", "prism": "triangle"}
    seen = {}   # shape -> (3D centre, lid z, camera yaw)
    H, W = depth.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    for shape_b, shape in name.items():
        h = lid.hole(shape_b)
        if h is None or shape not in out:
            continue   # hidden: placed below from the model layout (or kept, if no hole is seen at all)
        r = np.hypot(uu - h.u, vv - h.v)
        ring = (r > 28) & (r < 45) & (depth > 0)                  # plate around the hole (not through it)
        if ring.sum() < 30:
            report.append(f"lid: {shape} hole seen but no plate depth around it, expected position kept")
            continue
        P = cal.pixel_to_pelvis(cam, uu[ring], vv[ring], depth[ring].astype(float))
        z = float(np.median(P[:, 2]))
        # camera ray through the hole centre: two points at different depths
        a, b = cal.pixel_to_pelvis(cam, np.array([h.u, h.u]), np.array([h.v, h.v]), np.array([500.0, 1500.0]))
        d = b - a
        if abs(d[2]) < 1e-6:
            continue
        p = a + d * ((z - a[2]) / d[2])
        seen[shape] = (p, z, float(h.yaw))
    if not seen:
        return out, report
    # lid pose from the best seen hole (triangle first: its 120-deg yaw also gives the rotation) + the exact model layout
    ref = "triangle" if "triangle" in seen else next(iter(seen))
    p_ref, z_ref, yaw_ref = seen[ref]
    e_ref = np.asarray(expected[ref]["xy"], float)
    dth = 0.0
    if ref == "triangle" and expected[ref].get("dirs"):
        a_exp = float(np.arctan2(expected[ref]["dirs"][0][1], expected[ref]["dirs"][0][0]))   # expected apex direction
        dth = (yaw_ref - a_exp + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3                      # nearest of 3 (120 deg)
        dth = float(np.clip(dth, -np.radians(30), np.radians(30)))
        if abs(dth) < np.radians(3.0):   # below the camera's yaw noise (sim: 0.6-2.8 deg on an unturned lid)
            dth = 0.0
    c, s_ = np.cos(dth), np.sin(dth)
    R = np.array([[c, -s_], [s_, c]])
    for shape, exp in out.items():
        e = np.asarray(expected[shape]["xy"], float)
        if shape in seen:
            target, zt = seen[shape][0][:2], seen[shape][1]
            how = "camera"
        else:  # hidden: exact model offset from the reference hole, moved + turned like the reference
            target, zt = p_ref[:2] + R @ (e - e_ref), z_ref
            how = f"model from the {ref} hole"
        raw = 1000 * (target - e)
        dxy = np.clip(target - e, -tol, tol)
        dz = float(np.clip(zt - float(expected[shape]["z"]), -tol, tol))
        exp["xy"] = e + dxy
        exp["z"] = float(expected[shape]["z"]) + dz
        if exp.get("dirs"):
            exp["dirs"] = [R @ np.asarray(v, float) for v in expected[shape]["dirs"]]
        exp["camera"] = how
        report.append(f"lid: {shape} hole via {how}: offset ({raw[0]:+.0f},{raw[1]:+.0f}) mm -> applied "
                      f"({1000 * dxy[0]:+.0f},{1000 * dxy[1]:+.0f}) mm, z {1000 * dz:+.0f} mm, lid turn "
                      f"{np.degrees(dth):+.1f} deg (tol +-{1000 * tol:.0f} mm)")
    return out, report
