Wrote FINAL_TRAINING.md
# KITScenes Full-Set Training — Status & Plan

## Objective
Train the AutoE2E reactive part end-to-end on the full KITScenes Multimodal
dataset (1 epoch, resume from `checkpoints_full/best.pt`) using streaming
scene-tar downloads, deformable-attention map fusion, and a cosine LR schedule.
Validate on random single val tars per checkpoint.

## Split (plan 1a: size threshold ≥ 3.0 GB, metadata-only)
- `data/final_split.json` regenerated from `fs.info()` sizes.
- **train = 378** / 533 original train tars (155 filtered out, < 3.0 GB)
- **val = 70** / 117 original val tars (47 filtered out)
- All filtered IDs verified to exist in the SDK train / val splits.
- No scenes moved between splits (moved scenes caused a silent-skip bug).

## Key facts
- GPU: RTX 3060 12GB; PyTorch 2.4.1+cu124; **fp16 `torch.amp.autocast("cuda")`
  required** (fp32 OOMs). Batch=1, grad-accum=4.
- Deps: `kitscenes`, `lanelet2`, `scipy`. Data on `KIT-MRT/KITScenes-Multimodal`
  (gated; token via `HfFileSystem`).
- Size heuristic is UNRELIABLE for pose counting: `8b970653` 5.0GB→100 poses,
  `303a44ec` 4.7GB→100, `94740bfe` 4.7GB→110, `8494d673` 3.0GB→120,
  `57065161` 3.5GB→160. Plan 1a accepts this approximation.
- Streaming cache: 1–2 tars only; val tars evaluated then discarded;
  keep `best.pt` + `latest.pt` (overwrite) every 1–2 scenes.
- Cosine LR: 1e-4 → eta_min 1e-6, T_max ≈ total opt steps. AdamW, wd 1e-2.

## Fixes applied to `train_final.py`
1. `_download(sid, root, split)` extracts val tars to `root/val/{sid}`
   (was `root/train/{sid}`).
2. `_evict` scans both `root/train` and `root/val`.
3. `_validate_single` loads val with `split="val"` (was `split="train"`,
   which silently produced `n=0`).
4. `_clean_partial(root)` removes stale `.*.tar.part` at startup.
5. Every skipped scene is logged (load failure / 0 samples).
6. `best_ade`/`best.pt` update guarded: only when `vn > 0`.
7. `tar_log.txt` counts are dynamic (378 train / 70 val).

## Run command
```bash
cd Model/data_parsing/kit_scenes
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_final.py
```

## Prior ablation context
- 7-scene run: ADE 34.97→12.68m; 10-scene run: best ADE 4.32m;
  341-scene cached run: ADE 36.35→8.16m (abandoned).
- Deformable cross-attention fusion verified at full BEV 450×300 (135K tokens),
  K=4 learned offsets + `F.grid_sample`, 2.29GB, registered as `"deformable"`.

## Artifacts (git-ignored)
- `exp1/` — best.pt / latest.pt / history / tar_log.txt
- `exp2/`, `data/`, `trajectory_video*.mp4`
