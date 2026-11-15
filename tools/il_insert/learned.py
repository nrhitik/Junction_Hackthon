"""Learned final-stage controller (behaviour cloning of the sim expert): head RGB-D crop around the hole + low-dim
features (grasp point - hole, in-hand offset estimate, step) -> grasp-point move (dx, dy, dz) + release.

    ctrl = LearnedController("/data/il/cube/bc.pt")
"""
import numpy as np
import torch
import torch.nn as nn

MAX_STEP = 0.012


class BCNet(nn.Module):
    def __init__(self, n_low=7):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(64 + n_low, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 4))

    def forward(self, img, low):
        x = img.permute(0, 3, 1, 2).float() / 255.0
        h = torch.cat([self.cnn(x), low], 1)
        out = self.mlp(h)
        return torch.tanh(out[:, :3]) * MAX_STEP, out[:, 3]   # move (m), release logit


def low_scale(low):
    """grasp-point - hole in cm, in-hand offset in cm, valid flag, step fraction."""
    s = np.array([100, 100, 100, 100, 100, 1, 1], np.float32)
    return low * s


class LearnedController:
    def __init__(self, path, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=self.dev)
        self.net = BCNet(ck.get("n_low", 7)).to(self.dev)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.thr = float(ck.get("release_thr", 0.5))
        self.target = None

    @torch.no_grad()
    def __call__(self, feat, pol, obs):
        img = torch.as_tensor(feat["img"][None], device=self.dev)
        low = torch.as_tensor(low_scale(feat["low"])[None], device=self.dev)
        d, rel = self.net(img, low)
        return {"d": d[0].cpu().numpy().astype(float), "release": bool(torch.sigmoid(rel)[0] > self.thr)}
