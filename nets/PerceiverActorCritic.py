"""Grid-aware Perceiver actor-critic head."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from .transformer_utils import (
    Array,
    AttnConfig,
    DecoderBlock,
    PerceiverBlockConfig,
    PerceiverEncoderBlock,
    RoPEConfig,
    fourier_encode,
)


class GridEncoder(nn.Module):
    grid_size: int = 30
    latent_dim: int = 512
    num_cell_types: int = 16
    num_grid_types: int = 32
    fourier_encodings: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, grids: Array, grid_type_ids: Array, extra_feats: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Args:
            grids: (B, G, H, W) ints in [-1, num_cell_types-2]
            grid_type_ids: (B, G) or (G,) integers describing split/type
            extra_feats: optional (B, G, H, W, F) float features to fuse with tokens
        Returns:
            tokens: (B, G*H*W, latent_dim)
            pos_xy: (B, G*H*W, 2) absolute coordinates 0..grid_size-1
        """
        B, G, H, W = grids.shape
        assert H == self.grid_size and W == self.grid_size, "Grids must be 30x30"
        if grid_type_ids.ndim == 1:
            grid_type_ids = jnp.broadcast_to(grid_type_ids[None, :], (B, G))
        grid_type_ids = grid_type_ids.astype(jnp.int32)
        cell_ids = jnp.clip(grids + 1, 0, self.num_cell_types - 1).astype(jnp.int32)

        cell_embed = nn.Embed(
            self.num_cell_types,
            self.latent_dim,
            dtype=self.dtype,
            embedding_init=nn.initializers.xavier_uniform(),
            name="cell_embed",
        )(cell_ids)

        grid_type_embed = nn.Embed(
            self.num_grid_types,
            self.latent_dim,
            dtype=self.dtype,
            embedding_init=nn.initializers.xavier_uniform(),
            name="grid_type_embed",
        )(grid_type_ids)
        grid_type_embed = grid_type_embed[:, :, None, None, :]

        coords_hw = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.grid_size, dtype=jnp.float32),
                jnp.arange(self.grid_size, dtype=jnp.float32),
                indexing="ij",
            ),
            axis=-1,
        )  # (H, W, 2)
        coords = coords_hw[None, None, ...]
        coords = jnp.broadcast_to(coords, (B, G, H, W, 2))

        coords_norm = coords / (self.grid_size - 1.0)
        pos_feat = fourier_encode(coords_norm, self.fourier_encodings)
        pos_feat = rearrange(pos_feat, "b g h w p c -> b g h w (p c)").astype(self.dtype)
        pos_proj = nn.Dense(
            self.latent_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pos_proj",
        )(pos_feat)

        tokens = (cell_embed + grid_type_embed + pos_proj).astype(self.dtype)

        if extra_feats is not None:
            assert extra_feats.shape[:4] == (B, G, H, W), "Extra feats must align with grid layout"
            extra_projector = nn.Dense(
                self.latent_dim,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                name="extra_feat_proj",
            )
            extra_emb = extra_projector(extra_feats.astype(self.dtype))
            tokens = tokens + extra_emb

        tokens = rearrange(tokens, "b g h w c -> b (g h w) c")
        pos_xy = rearrange(coords[..., :2], "b g h w c -> b (g h w) c").astype(jnp.float32)
        return tokens, pos_xy


class PerceiverActorCritic(nn.Module):
    num_latents: int = 16
    latent_dim: int = 128
    depth: int = 2
    decoder_layers: int = 1
    ff_multiplier: int = 2
    dropout: float = 0.0
    token_dropout: float = 0.0
    num_heads: int = 2
    num_kv_heads: int = 2
    head_dim: int = 32
    dtype: jnp.dtype = jnp.bfloat16

    policy_dim: int = 20
    value_dim: int = 1

    num_cell_types: int = 16
    num_grid_types: int = 32
    canvas_type_id: int = 31

    input_rope_cfg: Optional[RoPEConfig] = None
    decoder_rope_cfg: Optional[RoPEConfig] = None

    def setup(self):
        self.task_grid_encoder = GridEncoder(
            latent_dim=self.latent_dim,
            num_cell_types=self.num_cell_types,
            num_grid_types=self.num_grid_types,
            dtype=self.dtype,
        )
        self.canvas_grid_encoder = GridEncoder(
            latent_dim=self.latent_dim,
            num_cell_types=self.num_cell_types,
            num_grid_types=self.num_grid_types,
            dtype=self.dtype,
        )

    def prepare_task_cache(self, task_grids: Array, task_grid_type_ids: Array) -> Tuple[Array, Array]:
        """Encode demonstrations once and reuse across multiple forward passes."""
        return self.task_grid_encoder(task_grids, task_grid_type_ids)

    @nn.compact
    def __call__(
        self,
        task_grids: Array,
        task_grid_type_ids: Array,
        canvas_grid: Array,
        extra_canvas_feats: Optional[Array] = None,
        cached_task_tokens: Optional[Array] = None,
        cached_task_pos: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Dict[str, Array]:
        """
        Args:
            task_grids: (B, G, 30, 30) all context grids
            task_grid_type_ids: (B, G) or (G,) identifiers (e.g. split/input-output/index)
            canvas_grid: (B, 30, 30) current canvas
            extra_canvas_feats: optional (B, 30, 30, F) float features per canvas cell
            cached_task_tokens: optional precomputed task tokens
            cached_task_pos: optional precomputed task positional encodings
        """
        B = canvas_grid.shape[0]

        if cached_task_tokens is None or cached_task_pos is None:
            task_tokens, task_pos = self.prepare_task_cache(task_grids, task_grid_type_ids)
        else:
            task_tokens, task_pos = cached_task_tokens, cached_task_pos

        canvas_ids = jnp.full((B, 1), self.canvas_type_id, dtype=jnp.int32)
        extra_canvas = None
        if extra_canvas_feats is not None:
            extra_canvas = extra_canvas_feats[:, None, ...]

        canvas_tokens, canvas_pos = self.canvas_grid_encoder(
            canvas_grid[:, None, :, :],
            canvas_ids,
            extra_canvas,
        )
        canvas_tokens = canvas_tokens.reshape(B, -1, self.latent_dim)
        canvas_pos = canvas_pos.reshape(B, -1, 2)

        latents = self.param(
            "latents",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.latent_dim),
        )
        latents = jnp.broadcast_to(latents[None, ...], (B,) + latents.shape).astype(self.dtype)

        attn_cfg = AttnConfig(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            use_qk_norm=True,
            dropout=self.token_dropout,
        )
        ff_hidden = self.latent_dim * self.ff_multiplier
        block_cfg = PerceiverBlockConfig(
            attn_cfg_cross=attn_cfg,
            attn_cfg_self=attn_cfg,
            ff_hidden=ff_hidden,
            ff_dropout=self.dropout,
        )

        for i in range(self.depth):
            latents = PerceiverEncoderBlock(
                block_cfg,
                rope_cfg_cross=self.input_rope_cfg,
                name=f"encoder_block_{i}",
            )(
                latents=latents,
                inputs=task_tokens,
                input_pos=task_pos,
                deterministic=deterministic,
            )

        decoder_tokens = jnp.concatenate([latents, canvas_tokens], axis=1)
        latent_pos = jnp.zeros((B, self.num_latents, 2), dtype=canvas_pos.dtype)
        decoder_pos = jnp.concatenate([latent_pos, canvas_pos], axis=1)

        decoder_attn_cfg = AttnConfig(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            use_qk_norm=True,
            dropout=self.dropout,
        )
        decoder_ff_hidden = self.latent_dim * self.ff_multiplier

        for i in range(self.decoder_layers):
            decoder_tokens = DecoderBlock(
                decoder_attn_cfg,
                self.decoder_rope_cfg,
                decoder_ff_hidden,
                self.dropout,
                name=f"decoder_block_{i}",
            )(decoder_tokens, decoder_pos, deterministic)

        decoded_latents = decoder_tokens[:, : self.num_latents]
        decoded_canvas = decoder_tokens[:, self.num_latents :]

        policy_logits = nn.Dense(
            self.policy_dim,
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
            name="policy_head",
        )(decoded_canvas.astype(self.dtype))

        value_tokens = decoded_latents.mean(axis=1)
        value = nn.Dense(
            self.value_dim,
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
            name="value_head",
        )(value_tokens.astype(self.dtype))
        if self.value_dim == 1:
            value = value.squeeze(-1)

        return {"logits": policy_logits, "value": value}
