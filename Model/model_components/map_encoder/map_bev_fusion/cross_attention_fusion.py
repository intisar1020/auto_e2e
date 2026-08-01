"""
Cross-attention map BEV fusion. Fuses image BEV features with map BEV features using spatial cross-attention.

"""

import torch
import torch.nn as nn


class MapCrossAttentionFusion(nn.Module):
    """Fuse image BEV and map BEV via spatial cross-attention.

    Args:
        embed_dim: Channel dimension of both input feature maps.
        num_heads: Number of attention heads.
        dropout: Dropout applied inside attention and the FFN.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_bev: torch.Tensor,
        map_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_bev: (B, embed_dim, H, W) image BEV features — queries.
            map_bev:   (B, embed_dim, H, W) map BEV features — keys and values.
                       Must have the same spatial size as image_bev.

        Returns:
            (B, embed_dim, H, W) image BEV updated with map context.
        """
        B, C, H, W = image_bev.shape

        # PyTorch 2.x uses flash attention / memory-efficient SDPA backends
        # which handle large token counts at O(N) memory, not O(N²).
        # The old guard is relaxed; if a particular GPU/backend still OOMs,
        # fall back to map_fusion_mode='residual'.

        # Flatten spatial dims: (B, H*W, C)
        q = image_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
        kv = map_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Pre-norm cross-attention with residual. need_weights=False avoids
        # materializing the full [B,heads,N,N] score tensor.
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(self.norm_query(q), kv_norm, kv_norm,
                                      need_weights=False)
        q = q + attn_out

        # FFN with residual
        q = q + self.ffn(self.norm_ffn(q))

        # Reshape back to spatial: (B, C, H, W)
        return q.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    