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
    def __call__(self, grids: Array, grid_type_ids: Array) -> Tuple[Array, Array]:
        """
        Args:
            grids: (B, G, H, W) ints in [-1, num_cell_types-2]
            grid_type_ids: (B, G) or (G,) integers describing split/type
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
        tokens = rearrange(tokens, "b g h w c -> b (g h w) c")
        pos_xy = rearrange(coords[..., :2], "b g h w c -> b (g h w) c").astype(jnp.float32)
        return tokens, pos_xy


class PerceiverActorCritic(nn.Module):
    num_latents: int = 64
    latent_dim: int = 512
    depth: int = 6
    decoder_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.0
    token_dropout: float = 0.0
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 64
    dtype: jnp.dtype = jnp.bfloat16

    policy_dim: int = 20
    value_dim: int = 1

    num_cell_types: int = 16
    num_grid_types: int = 32
    canvas_type_id: int = 31

    input_rope_cfg: Optional[RoPEConfig] = None
    decoder_rope_cfg: Optional[RoPEConfig] = None

    @nn.compact
    def __call__(
        self,
        task_grids: Array,
        task_grid_type_ids: Array,
        canvas_grid: Array,
        deterministic: bool = True,
    ) -> Dict[str, Array]:
        """
        Args:
            task_grids: (B, G, 30, 30) all context grids
            task_grid_type_ids: (B, G) or (G,) identifiers (e.g. split/input-output/index)
            canvas_grid: (B, 30, 30) current canvas
        """
        B = task_grids.shape[0]
        encoder = GridEncoder(
            latent_dim=self.latent_dim,
            num_cell_types=self.num_cell_types,
            num_grid_types=self.num_grid_types,
            dtype=self.dtype,
            name="grid_encoder",
        )
        task_tokens, task_pos = encoder(task_grids, task_grid_type_ids)

        canvas_ids = jnp.full((B, 1), self.canvas_type_id, dtype=jnp.int32)
        canvas_tokens, canvas_pos = encoder(canvas_grid[:, None, :, :], canvas_ids)
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
