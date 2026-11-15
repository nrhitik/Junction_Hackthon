"""Hole outlines straight from the lid STL: slice at mid-thickness, report outer outline + each hole (centre, size,
shape guess) in the STL's own units/frame, and relative to the outline centre."""
import sys
import numpy as np
import trimesh

for path in sys.argv[1:]:
    m = trimesh.load(path, force="mesh")
    lo, hi = m.bounds
    print(f"== {path}\n   bounds {lo.round(2)} .. {hi.round(2)}  (extent {(hi - lo).round(2)})")
    ax = int(np.argmin(hi - lo))  # thin axis = lid normal
    origin = np.zeros(3); origin[ax] = 0.5 * (lo[ax] + hi[ax])
    normal = np.zeros(3); normal[ax] = 1
    sec = m.section(plane_origin=origin, plane_normal=normal)
    planar, T = sec.to_2D()
    polys = planar.polygons_full
    for P in polys:
        oc = np.asarray(P.exterior.centroid.coords[0])
        ob = np.asarray(P.exterior.bounds)
        # back to 3D STL coords for the outline centre
        c3 = trimesh.transform_points(np.c_[[oc], [0]], T)[0]
        print(f"   outline centre(3D) {c3.round(2)} size {(ob[2:] - ob[:2]).round(2)} holes {len(P.interiors)} thin axis {'xyz'[ax]}")
        for I in P.interiors:
            from shapely.geometry import Polygon
            h = Polygon(I)
            hc = np.asarray(h.centroid.coords[0])
            hb = np.asarray(h.bounds)
            n = len(h.simplify(0.02 * max(hb[2] - hb[0], 1e-9)).exterior.coords) - 1
            circ = 4 * np.pi * h.area / h.length ** 2
            guess = "circle" if circ > 0.85 else ("triangle" if n <= 4 or circ < 0.65 else "square")
            h3 = trimesh.transform_points(np.c_[[hc], [0]], T)[0]
            print(f"     hole {guess:8s} centre(3D) {h3.round(2)} rel outline centre {(h3 - c3).round(2)} "
                  f"bbox {(hb[2:] - hb[:2]).round(2)} area {h.area:.1f} circularity {circ:.2f}")
