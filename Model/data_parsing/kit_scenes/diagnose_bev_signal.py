"""Diagnose whether the fused BEV carries scene content.

Measures, at each stage of the image BEV pipeline, the correlation between the
feature maps of two DIFFERENT scenes (same model, no grad). scene corr -> 1.0
means the stage is scene-INVARIANT (broken); much lower means the stage carries
scene-dependent structure.

Also runs the camera-swap functional test: swap scene B's camera tiles into
scene A's forward and measure how much the output trajectory changes.

Usage:
  python diagnose_bev_signal.py [--ckpt path] [--image-feature-size 32] [--gate]
"""

import sys
import argparse
import importlib.util

import numpy as np
import torch

sys.path.insert(0, "/home/intisar/Documents/source/auto_e2e/Model")
KIT = "/home/intisar/Documents/source/auto_e2e/Model/data_parsing/kit_scenes"
sys.path.insert(0, KIT)

spec = importlib.util.spec_from_file_location("tm", KIT + "/train_main.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
tm.ROUTE_ONLY = True

from data_parsing.kit_scenes import KitScenesDataset, NUM_VIEWS
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion.projection import PinholeProjection


def build_model(image_feature_size, ckpt=None, device="cuda"):
    model = AutoE2E(
        enable_reasoning=False,
        num_views=NUM_VIEWS,
        map_context_channels=0,
        route_channels=2,
        map_fusion_mode="deformable",
        image_feature_size=image_feature_size,
    ).to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def stage_corr(model, val_ds, device):
    """scene-correlation at each stage across two different scenes."""
    f = model.Reactive_E2E.FeatureFusion
    view = f.view_fusion
    caps = {}

    orig_ff = f.forward
    def patched_ff(features, B, V, projection=None, geometry_type=None,
                   image_transform=None):
        for i in range(len(features)):
            features[i] = f.pool(features[i])
        fpv = torch.cat(features, dim=1)
        fpv = f.channel_proj(fpv)
        caps.setdefault("fpv", []).append(fpv.detach().float().cpu())
        return view(fpv, B, V, projection=projection, geometry_type=geometry_type,
                    image_transform=image_transform)
    f.forward = patched_ff

    hook = view.register_forward_hook(
        lambda mod, args, out: caps.setdefault("image_bev", []).append(
            out.detach().float().cpu()
        )
    )

    for idx in (0, 1):
        s = val_ds[idx]
        v, eg, vh, tg, cp, mc, rm, mv, rv = tm._tensors(s, device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                      route_valid=rv, projection=PinholeProjection(cp),
                      geometry_type="pinhole", trajectory_target=tg,
                      mode="infer")
    f.forward = orig_ff
    hook.remove()

    def corr(a, b):
        a = a.reshape(a.shape[0], -1).numpy()
        b = b.reshape(b.shape[0], -1).numpy()
        return float(np.mean([
            np.corrcoef(a[c], b[c])[0, 1]
            for c in range(0, a.shape[0], max(1, a.shape[0] // 16))
        ]))

    print("stage-wise scene correlation (two different scenes; 1.0 = invariant):")
    for key, label in (("fpv", "per-view features (backbone+proj)"),
                       ("image_bev", "image_bev (after view fusion)")):
        a, b = caps[key]
        print(f"  {label:35s} corr={corr(a[0], b[0]):.3f}")
    return caps


def camera_swap(model, val_ds, device):
    """Functional test: how much does the output change with camera content?

    NOTE: tensors are rebuilt fresh per forward because AutoE2E normalizes
    egomotion IN-PLACE (divides speed/accel by 33/8); reusing a tensor across
    calls double-normalizes and corrupts the measurement.
    """
    sA, sB = val_ds[0], val_ds[1]
    tA = tm._tensors(sA, device)
    vA, egA, vhA, tgA, cpA, mcA, rmA, mvA, rvA = tA
    vB = tm._tensors(sB, device)[0]

    def run(v, eg, cp):
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                return model(v, mcA, vhA, eg, route_mask=rmA, map_valid=mvA,
                             route_valid=rvA, projection=PinholeProjection(cp),
                             geometry_type="pinhole", trajectory_target=tgA,
                             mode="infer")

    traj_AA = run(vA, egA, cpA)
    # Fresh eg for each run (in-place egomotion normalization would corrupt).
    traj_AB = run(vB, tm._tensors(sA, device)[1], cpA)   # B images on A ego/calib
    traj_Az = run(torch.zeros_like(vA), tm._tensors(sA, device)[1], cpA)
    print("camera-swap functional test (mean |d| on controls):")
    print(f"  |A - A|                  : {0.0:.4f}  (baseline)")
    print(f"  |A - B-images-on-A|      : {(traj_AA - traj_AB).abs().mean().item():.4f}")
    print(f"  |A - zero-images|        : {(traj_AA - traj_Az).abs().mean().item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--image-feature-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    vds = KitScenesDataset(data_root=str(tm.DATA_ROOT), split="train",
                           include_navigation=True, scene_ids=tm._load_split()[1][:2])
    model = build_model(args.image_feature_size, ckpt=args.ckpt, device=args.device)
    gate = model.Reactive_E2E.FeatureFusion.view_fusion.image_gate.item()
    print(f"image_feature_size={args.image_feature_size}  image_gate={gate}  "
          f"ckpt={args.ckpt}")
    stage_corr(model, vds, args.device)
    camera_swap(model, vds, args.device)


if __name__ == "__main__":
    main()
