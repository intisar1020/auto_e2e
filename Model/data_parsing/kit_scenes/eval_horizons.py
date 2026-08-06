"""Compute ADE/FDE at 3s and 6s horizons for the best checkpoint on the val set.

Trajectory = 64 steps @ 10Hz = 6.4s -> 3s = 30 steps, 6s = 60 steps.

Usage:
  cd Model/data_parsing/kit_scenes
  python eval_horizons.py [--ckpt exp-1-subset/checkpoints/best.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation

BASE = Path(__file__).parent / "exp-1-subset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(BASE / "checkpoints/best.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    m = json.loads((BASE / "manifest.json").read_text())
    val_ids = [e["sid"] for e in m["tars"]
               if e["role"] == "val" and e.get("status") == "done"]
    vds = KitScenesDataset(data_root=str(BASE / "data"), split="train",
                           include_navigation=True, scene_ids=val_ids)
    print(f"val: {len(vds)} samples / {len(val_ids)} tars")

    model = AutoE2E(enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    n_steps = 64  # 10Hz -> 6.4s
    v0 = 10.0
    acc_e = np.zeros((len(vds), n_steps))
    fde_at = {}
    with torch.no_grad():
        for i in range(len(vds)):
            s = vds[i]
            v = s["visual_tiles"].unsqueeze(0).to(device).float() / 255.0
            eg = s["egomotion_history"].unsqueeze(0).to(device).float()
            vh = s["visual_history"].unsqueeze(0).to(device).float()
            tg = s["trajectory_target"].unsqueeze(0).to(device).float()
            cp = s["camera_params"].unsqueeze(0).to(device).float()
            mc, rm, _ = decode_sample_navigation(s["navigation_members"])
            mc = torch.from_numpy(mc.copy()).float().unsqueeze(0).to(device)
            rm = torch.from_numpy(rm.copy()).float().unsqueeze(0).to(device)
            mv = torch.ones(1, dtype=torch.bool, device=device)
            rv = torch.zeros(1, dtype=torch.bool, device=device)
            with torch.amp.autocast("cuda"):
                out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                            route_valid=rv, projection=PinholeProjection(cp),
                            geometry_type="pinhole", trajectory_target=tg,
                            mode="train")
            pred = out.detach().cpu().numpy()[0]
            tgt = tg.cpu().numpy()[0]
            px = integrate_trajectory(pred[0::2], pred[1::2], v0=v0)
            tx = integrate_trajectory(tgt[0::2], tgt[1::2], v0=v0)
            er = np.linalg.norm(px - tx, axis=1)
            acc_e[i] = er

    for horizon_s in (3, 6):
        k = int(horizon_s * 10)  # 30, 60
        ade = acc_e[:, :k].mean(axis=1).mean()
        fde = acc_e[:, k - 1].mean()
        print(f"{horizon_s}s: ADE {ade:.4f} m | FDE {fde:.4f} m")

    ade_full = acc_e.mean()
    fde_full = acc_e[:, -1].mean()
    print(f"full (6.4s): ADE {ade_full:.4f} m | FDE {fde_full:.4f} m")


if __name__ == "__main__":
    main()
