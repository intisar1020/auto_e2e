# [Investigation] Fused BEV features are scene-invariant: PCA visualization shows only noise (no spatial pattern)

## Symptom

After 1 epoch of training (`exp-5-route-only`, from scratch, route-only 2-ch navigation, imitation loss), the per-epoch PCA visualization of the **fused image+map BEV features** (before `FusedFeaturePooling`) looks like **random jitter of colors on gray — no spatial pattern at all**. Even across training epochs the structure does not emerge.

This is the fused BEV map produced by `MapBEVFusion(image_bev, navigation_bev)`, shape `(B, 256, 450, 300)`, i.e. **before** the channel-compression/pooling stage.

## How the PCA visualization is produced

Per epoch, during validation:

1. Register a `forward_hook` on `model.Reactive_E2E.MapBEVFusion` to capture its output `fused_features` (shape `(B, embed_dim, bev_h, bev_w)` = `(1, 256, 450, 300)`).
2. Run a small fixed subset of val samples through the model in `infer` mode (`torch.no_grad()` + fp16 autocast).
3. Concatenate captured features along the batch axis → `(N, 256, 450, 300)`.
4. Flatten to `(N*450*300, 256)` and fit `sklearn.decomposition.PCA(n_components=3)`.
5. Project back → `(N, 450, 300, 3)`, per-sample min-max normalize to `[0,1]`.
6. Render one RGB image per sample in a row, save to `exp-<name>/pca_bev/epoch_<epoch>.png`, with the PCA explained-variance ratios in the title.

A healthy BEV should show coherent spatial structure (road/lane/route layout, camera coverage seams, scene geometry). What we see instead is per-pixel random color, i.e. **the BEV carries no scene structure**.

## Root-cause analysis

### 1. Camera projection is CORRECT (not the 6-camera encoding)

Instrumenting `PinholeProjection` on real calibration:

- A point 20 m ahead, 0 m lateral → front-center camera image center.
- 20 m ahead, +3 m left → front-center image shifted left; −3 m right → shifted right.
- 10 m behind → rear camera.

Coverage: **100% of BEV cells see at least one camera**; per-camera coverage 15–49% (sensible for a 6-cam surround). All 64 coarse per-view cells are hit per camera.

### 2. Per-view image features ARE scene-dependent

Capturing the per-view features after backbone + `channel_proj` (shape `(6, 256, 8, 8)`), the correlation between **two different scenes** at the same pixel/channel is:

```
cam0 0.700  cam1 0.777  cam2 0.368  cam3 0.794  cam4 0.835  cam5 0.405
```

So the cameras genuinely encode scene content and the features differ between scenes.

### 3. But the fused BEV is scene-INVARIANT

Measuring the correlation between the fused BEV of two **completely different scenes** (same model, same checkpoints, different samples):

| Stage | scene corr | std |
|---|---|---|
| per-view features (after backbone+proj) | 0.37–0.84 | 0.37 |
| sampled image signal (`weighted_avg`, before query residual) | **0.994** | 0.19 |
| after `output_proj(sampled)` | 0.994 | 0.08 |
| `image_bev` (queries + residual, after LayerNorm+FFN) | **1.000** | 0.98 |
| `fused_features` (after `MapBEVFusion`) | **1.000** | 0.98 |

scene corr = 1.0 means the two scenes produce **identical** BEV tensors. The scene content is completely lost.

### 4. Functional proof (cameras present vs. content)

Comparing trajectory outputs (mean abs diff on `(1,128)` controls):

```
|A - A|                    : 0.0000  (baseline)
|A - B-images-on-A|        : 0.0278  (swap in a different scene's cameras)
|A - zero-images|          : 0.2427  (zero the cameras entirely)
|A - B (full swap)         : 0.0398
```

The model reacts strongly to **whether cameras are present** (0.24) but barely to **what is in them** (0.028). I.e. the image path contributes "something is there", not the actual scene.

## Two compounding causes

### A. `image_feature_size=8` over-compression bottleneck

`FeatureFusion` (`Model/model_components/feature_fusion.py`) pools the backbone's multi-scale feature maps (64×64, 32×32, 16×16, 8×8) with `AdaptiveMaxPool2d(8)` before BEV sampling. The whole 256×256 image is therefore reduced to **8×8 = 64 spatial cells per camera** before `BEVViewFusion` samples from it. That is an extreme information loss — every 8×8 cell represents a 32×32 pixel region averaged down to one vector.

### B. Fixed per-cell query embedding dominates the residual

In `BEVViewFusion` (`Model/model_components/view_fusion/bev_fusion.py`):

```python
output = queries + self.output_proj(output)   # queries: learned per-BEV-cell embedding
output = self.norm1(output)
output = output + self.ffn(output)
output = self.norm2(output)
```

- `queries` (the learned `bev_queries` embedding, std ≈ **0.99**) is the same for every scene.
- the projected image signal `output_proj(sampled)` has std ≈ **0.08** — ~12× smaller.

After the residual + LayerNorm the output is ~99% the fixed query pattern and ~1% image content → the BEV is effectively a constant per-cell function of the input. PCA of a near-constant-per-cell tensor is exactly "random jitter of colors on gray".

Note: at random init the sampled image signal std is larger (~1.1), so the phenomenon is **architectural / scaling**, not a training-collapse artifact. A fresh untrained model already yields scene corr ≈ 0.98.

## Summary

The 6-camera encoding (calibration, projection, backbone) is **not** broken — geometry is verified and per-view features are scene-dependent. The loss of scene information happens in the BEV construction:

1. images are compressed to 8×8 per camera before BEV sampling, and
2. the fixed per-cell query embedding drowns the (weak) image signal in the `queries + output_proj(...)` residual.

Result: the fused BEV is scene-invariant, so the planner receives ~no image-derived spatial structure, and the PCA visualization correctly shows noise.

## Fix implemented (exp-6)

Two changes in `Model/model_components/`:

1. **`image_feature_size` 8 → 32** (`feature_fusion.py`, `auto_e2e.py`, `reactive_e2e.py`; exposed as `--image-feature-size` in `train_main.py`). Per-view value map goes from 64 → 1024 cells/camera before BEV sampling. Compute cost: **+6% wall time, +1% VRAM** (measured: 192→204 ms fwd, 2747→2777 MB peak at batch 1) — the dominant `grid_sample` output is fixed by the BEV grid, not per-view resolution. Same batch size works.

2. **Re-balanced the query residual** (`view_fusion/bev_fusion.py`):
   - `bev_queries` init std **1.0 → 0.1** (the fixed per-cell embedding was ~28× louder than the image signal; at 0.1 the image signal is 3.6× stronger).
   - `image_gate` init **0 → 1.0** on `output = queries + image_gate * output_proj(sampled)`. A zero-init gate gave `d(loss)/d(output_proj) ≈ gate = 0`, i.e. **no gradient to the image path** (measured: gate grad −4.7e-4, output_proj grad 0.0 after 2 epochs) — the branch never activates. Init 1.0 keeps the image signal live from step 0.

Result on a **fresh, untrained** model:

| stage | before fix | after fix |
|---|---|---|
| `image_bev` scene corr | 1.000 | **0.644** |
| `fused_features` scene corr | 1.000 | scene-dependent |

Training (exp-6, from scratch, 50 epochs): epoch 1 **ADE 6.30 / FDE 17.18** (vs exp-5 epoch-1 11.42 / 28.04); 3s ADE 1.75. PCA `pca_bev/epoch_*.png` now shows real color structure (33k unique RGB per frame) instead of the gray-jitter noise.

**Note on the camera-swap test:** the earlier "|A−B-images| = 0.156" was a measurement artifact — `AutoE2E.normalize_egomotion` mutates its input in place, and the diagnostic reused the same `eg` tensor, double-normalizing it. `diagnose_bev_signal.py` now rebuilds tensors per forward; the test is deterministic (baseline |A−A| = 0.0000).

## Follow-up: LayerNorm was stripping the per-cell scene signal

After the exp-6 fix, the PCA became colorful but still looked the same across scenes. Tracing the image-swap sensitivity (rel change of the BEV feature map when swapping one scene's cameras for another's) through the `BEVViewFusion` forward showed the exact culprit:

| stage | image-swap rel change (before) |
|---|---|
| raw sampled image features | 6.3% |
| after attention weighting | 6.3% |
| residual `queries + gate·output_proj` | 5.0% |
| **after `norm1` LayerNorm** | **1.4%** |

The scene signal lives in the per-cell **mean** (8.2% rel change) and **std** (4.8%), and the post-residual `nn.LayerNorm` normalizes each BEV cell independently — removing exactly those statistics. This is the classic "LayerNorm strips global/DC scene statistics" failure.

**Fix:** removed both post-residual LayerNorms (`norm1`, `norm2`) and switched to a pre-norm FFN (`output + ffn(norm_ffn(output))`), matching the deformable map fusion's pattern. After this, on the trained model the scene signal survives to the output:

| stage | image-swap rel change (after) |
|---|---|
| `values` (value_proj of per-view) | 53.3% |
| `weighted` (grid_sample + attention) | 29.5% |
| residual `queries + gate·output_proj` | 16.8% |
| `ffn_out` (final image_bev) | **16.3%** (was 1.2%) |

**Remaining dilution** (not yet addressed):
1. `values → weighted` (53% → 30%): grid_sample + per-cell attention-weight aggregation over 4 pillar z-points + camera-averaging `1/|V_hit|` mixes scene content.
2. `weighted → residual` (30% → 17%): adding the (scene-invariant) fixed query and `output_proj`.

**Also a training-objective tension:** the imitation target is a continuation of the egomotion history (speed/accel/yaw/curvature), so the model can minimize imitation loss largely from egomotion + route alone; the image path has healthy gradients now (`output_proj.grad=1.1e-3`, `gate.grad=0.058`) but is not strictly required by the loss.

## Suggested next steps

- Let exp-6 run to convergence (50 epochs); track ADE vs exp-2/3 baselines.
- Optionally ablate `image_feature_size` (16 vs 32) and gate init (0.1 vs 1.0) to attribute gains.
- Verify with the scene-correlation and camera-swap functional tests above (they are cheap and decisive).
