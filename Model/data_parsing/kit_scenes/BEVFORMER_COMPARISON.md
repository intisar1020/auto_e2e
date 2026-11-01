# BEVFormer-style trajectory planner vs. OG reactive BEVFormer

Comparison between the new BEVFormer-style encoder (`bevformer_pure_torch.py`,
`BEVFormerTrajectoryNet`) and the OG reactive "BEVFormer" (`view_fusion/bev_fusion.py`,
`BEVViewFusion` used by `AutoE2E` / `ReactiveE2E`).

---

## OG reactive "BEVFormer" — `BEVViewFusion`

A **simplified single-layer, single-head, single-scale** spatial cross-attention,
described in its own docstring as "simplified variant inspired by BEVFormer's
spatial cross-attention" that "does not fully replicate BEVFormer's multi-head
deformable attention":

1. **One fusion pass, no encoder stack.** Learnable BEV queries
   (`450×300` = 135k cells) sample camera features once.
2. **Single head, single level.** `sampling_offsets` =
   `Linear(embed_dim → num_points*2)`, `attention_weights` =
   `Linear(embed_dim → num_points)`. There is **no `num_heads` axis** and
   **no multi-scale axis** (`spatial_shapes` / `level_start_index` don't exist).
   Values are one tensor: all backbone scales pooled to `image_feature_size`
   (8/32) then channel-reduced and concatenated.
3. **No temporal self-attention.** No prev-BEV → current-BEV. Temporal context
   flows only through the separate `TemporalMemory` (visual history + egomotion
   → planner context vectors).
4. **Static reference points.** Precomputed 3D pillars, projected per camera; no
   egomotion-based BEV grid shift/alignment.
5. **Post-norm residual FFN:** `output = queries + output_proj(sample)` →
   `norm1` → `+ffn` → `norm2`.
6. **Head crushes the BEV to a vector:** `FusedFeaturePooling` (Conv2d 256→1 +
   AdaptiveAvgPool2d 75×50 → flatten to **3750**) → concat ego+visual history →
   `BezierPlanner` (MLP → 5 control points → Bernstein basis → 64×2).

---

## New BEVFormer — `BEVFormerTrajectoryNet`

A faithful, full BEVFormer (ECCV 2022) port in pure PyTorch, minus the detection
head:

1. **Stacked encoder**: `num_encoder_layers` (default 2) ×
   `PureBEVFormerEncoderLayer`, each = temporal self-attn → spatial cross-attn →
   FFN.
2. **Multi-head, multi-scale deformable attention** (`PureMSDeformableAttention`):
   `sampling_offsets` `Linear(embed_dim → H*L*P*2)`, `attention_weights`
   `Linear(embed_dim → H*L*P)`, softmax over levels×points, `grid_sample`-based
   `_multi_scale_deformable_attn` with `spatial_shapes` / `level_start_index`, and
   the original circular/spiral sampling init. `BEVFormerImageEncoder` keeps the
   backbone's real multi-level features (each projected to embed_dim, plus camera
   and level embeddings).
3. **Temporal self-attention** (`PureTemporalSelfAttention`): current BEV queries
   attend to the previous BEV (2-beat queue), with a **shift estimated from last
   ego speed** for grid alignment. This is BEVFormer's signature feature — absent
   in the OG.
4. **BEV grid 64×64** (4096 queries), queries init `N(0, 0.02)` + learned
   `bev_pos`.
5. **Spatial cross-attention averages over visible cameras** (`1/|V_hit|`) with
   residual, mirroring the original SCA.
6. **Head preserves the BEV**: `QueryPlanner` runs learned 5 control queries as
   cross-attention over the **full flattened BEV `[B, H*W, C]`** (no channel crush
   to 1), conditioned on ego+visual context, outputs control points → Bernstein
   basis → 64×2.
7. Map: small `BEVFormerMapEncoder` (conv net) + `BEVFormerMapFusion`
   (concat → conv → add) instead of the shared `NavigationEncoder`.

---

## Summary table

| | OG reactive `BEVViewFusion` | New `BEVFormerTrajectoryNet` |
|---|---|---|
| Encoder depth | 1 layer | 2 stacked layers |
| Heads | single-head | multi-head |
| Scales | 1 (pooled concat) | multi-scale deformable |
| Temporal self-attn | none | yes (+ ego-speed shift) |
| BEV grid | 450×300 (135k) | 64×64 (4k) |
| Head | crush→3750→MLP Bezier | query cross-attn over full BEV |
| Detection head | no | no |
