"""Shared attention utilities for Perceiver-style models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

Array = jax.Array


def fourier_encode(x: Array, num_encodings: int = 4) -> Array:
    """Sin/cos Fourier features plus raw coordinates."""
    x = jnp.expand_dims(x, -1)
    orig = x
    scales = (2.0 ** jnp.arange(num_encodings)).astype(x.dtype)
    x = x / scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    return jnp.concatenate([x, orig], axis=-1)


class RMSNorm(nn.Module):
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = x.astype(self.dtype)
        x32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), dtype=self.dtype)
        return (x32 / rms).astype(self.dtype) * scale


def qk_norm(q: Array, k: Array, eps: float = 1e-6) -> Tuple[Array, Array]:
    def _norm(a):
        a32 = a.astype(jnp.float32)
        var = jnp.mean(jnp.square(a32), axis=-1, keepdims=True)
        return (a32 * jax.lax.rsqrt(var + eps)).astype(a.dtype)

    return _norm(q), _norm(k)


def swiglu(x: Array, limit: float = 7.0, alpha: float = 1.702) -> Array:
    x_glu, x_lin = jnp.split(x, 2, axis=-1)
    x_glu = jnp.clip(x_glu, a_min=None, a_max=limit)
    x_lin = jnp.clip(x_lin, a_min=-limit, a_max=limit)
    return (x_glu * jax.nn.sigmoid(alpha * x_glu)) * (x_lin + 1.0)


@dataclass
class RoPEConfig:
    head_dim: int
    base_theta: float = 10000.0
    initial_context: int = 4096
    scaling_factor: float = 1.0
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0


def _rope_concentration_and_inv_freq(cfg: RoPEConfig):
    d = cfg.head_dim
    assert d % 2 == 0, "RoPE head_dim must be even"
    freq = cfg.base_theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d)
    if cfg.scaling_factor > 1.0:
        concentration = 0.1 * jnp.log(cfg.scaling_factor) + 1.0
        low = (d // 2) * jnp.log(cfg.initial_context / (cfg.ntk_beta * 2 * jnp.pi)) / jnp.log(cfg.base_theta)
        high = (d // 2) * jnp.log(cfg.initial_context / (cfg.ntk_alpha * 2 * jnp.pi)) / jnp.log(cfg.base_theta)
        ramp = (jnp.arange(d // 2, dtype=jnp.float32) - low) / (high - low)
        mask = 1.0 - jnp.clip(ramp, 0.0, 1.0)
        inv_freq = (1.0 / (cfg.scaling_factor * freq)) * (1.0 - mask) + (1.0 / freq) * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq
    return concentration, inv_freq


def _rope_rotate(x: Array, pos: Optional[Array], cfg: RoPEConfig) -> Array:
    if pos is None:
        return x
    conc, inv_freq = _rope_concentration_and_inv_freq(cfg)
    pos = pos.astype(jnp.float32)
    freqs = pos[..., None] * inv_freq[None, ...]
    cos, sin = jnp.cos(freqs) * conc, jnp.sin(freqs) * conc
    cos = cos[..., None, :].astype(x.dtype)
    sin = sin[..., None, :].astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def apply_2d_rope(
    q: Array,
    k: Array,
    pos_q: Optional[Array],
    pos_k: Optional[Array],
    cfg: RoPEConfig,
) -> Tuple[Array, Array]:
    assert q.shape[-1] == k.shape[-1], "q/k head_dim mismatch"
    head_dim = q.shape[-1]
    assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
    axis_dim = head_dim // 2
    axis_cfg = replace(cfg, head_dim=axis_dim)

    def _split_xy(x):
        return jnp.split(x, 2, axis=-1)

    qx, qy = _split_xy(q)
    kx, ky = _split_xy(k)
    pos_q_x = pos_q[..., 0] if pos_q is not None else None
    pos_q_y = pos_q[..., 1] if pos_q is not None else None
    pos_k_x = pos_k[..., 0] if pos_k is not None else None
    pos_k_y = pos_k[..., 1] if pos_k is not None else None

    qx = _rope_rotate(qx, pos_q_x, axis_cfg)
    qy = _rope_rotate(qy, pos_q_y, axis_cfg)
    kx = _rope_rotate(kx, pos_k_x, axis_cfg)
    ky = _rope_rotate(ky, pos_k_y, axis_cfg)
    return jnp.concatenate([qx, qy], axis=-1), jnp.concatenate([kx, ky], axis=-1)


def repeat_kv(kv: Array, n_rep: int) -> Array:
    if n_rep == 1:
        return kv
    return rearrange(kv[:, :, :, None, :].repeat(n_rep, axis=3), "b t hkv r d -> b t (hkv r) d")


class SwiGLUFFN(nn.Module):
    hidden_size: int
    out_size: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array, deterministic: bool) -> Array:
        h = nn.Dense(
            self.hidden_size * 2,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        h = swiglu(h.astype(self.dtype))
        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(
            self.out_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(h)


@dataclass
class AttnConfig:
    num_heads: int
    head_dim: int
    num_kv_heads: int
    dtype: jnp.dtype = jnp.bfloat16
    use_qk_norm: bool = True
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    cfg: AttnConfig
    rope_cfg: Optional[RoPEConfig] = None

    @nn.compact
    def __call__(
        self,
        x_q: Array,
        x_kv: Optional[Array] = None,
        mask: Optional[Array] = None,
        attn_bias: Optional[Array] = None,
        q_pos: Optional[Array] = None,
        k_pos: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Array:
        cfg = self.cfg
        x_kv = x_q if x_kv is None else x_kv
        H, D = cfg.num_heads, cfg.head_dim
        Hkv = cfg.num_kv_heads
        assert H % Hkv == 0, "num_heads must be multiple of num_kv_heads"
        rep = H // Hkv

        q = nn.DenseGeneral(
            (H, D),
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="q",
        )(x_q)
        k = nn.DenseGeneral(
            (Hkv, D),
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="k",
        )(x_kv)
        v = nn.DenseGeneral(
            (Hkv, D),
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="v",
        )(x_kv)

        if cfg.use_qk_norm:
            q, k = qk_norm(q, k)

        k = repeat_kv(k, rep)
        v = repeat_kv(v, rep)
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

        if self.rope_cfg is not None and (q_pos is not None or k_pos is not None):
            q, k = apply_2d_rope(q, k, q_pos, k_pos, self.rope_cfg)

        scale = jnp.asarray(1.0 / jnp.sqrt(D), dtype=jnp.float32)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if attn_bias is not None:
            logits = logits + attn_bias
        if mask is not None:
            logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        attn = nn.softmax(logits, axis=-1)
        attn = nn.Dropout(cfg.dropout)(attn, deterministic=deterministic)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).astype(cfg.dtype)
        out = rearrange(out, "b q h d -> b q (h d)")
        return nn.Dense(
            x_q.shape[-1],
            use_bias=False,
            dtype=cfg.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="o",
        )(out)


@dataclass
class PerceiverBlockConfig:
    attn_cfg_cross: AttnConfig
    attn_cfg_self: AttnConfig
    ff_hidden: int
    ff_dropout: float = 0.0


class PerceiverEncoderBlock(nn.Module):
    cfg: PerceiverBlockConfig
    rope_cfg_cross: Optional[RoPEConfig] = None

    @nn.compact
    def __call__(
        self,
        latents: Array,
        inputs: Array,
        input_pos: Optional[Array],
        deterministic: bool,
        input_mask: Optional[Array] = None,
    ) -> Array:
        x = RMSNorm(dtype=self.cfg.attn_cfg_cross.dtype)(latents)
        x = MultiHeadAttention(self.cfg.attn_cfg_cross, self.rope_cfg_cross, name="cross_attn")(
            x_q=x,
            x_kv=inputs,
            q_pos=None,
            k_pos=input_pos,
            mask=input_mask,
            deterministic=deterministic,
        )
        latents = latents + x

        y = RMSNorm(dtype=self.cfg.attn_cfg_cross.dtype)(latents)
        y = SwiGLUFFN(
            self.cfg.ff_hidden,
            latents.shape[-1],
            dropout_rate=self.cfg.ff_dropout,
            dtype=self.cfg.attn_cfg_cross.dtype,
            name="ffn",
        )(y, deterministic)
        latents = latents + y

        z = RMSNorm(dtype=self.cfg.attn_cfg_self.dtype)(latents)
        z = MultiHeadAttention(self.cfg.attn_cfg_self, name="latent_self_attn")(
            x_q=z,
            x_kv=None,
            deterministic=deterministic,
        )
        latents = latents + z

        t = RMSNorm(dtype=self.cfg.attn_cfg_self.dtype)(latents)
        t = SwiGLUFFN(
            self.cfg.ff_hidden,
            latents.shape[-1],
            dropout_rate=self.cfg.ff_dropout,
            dtype=self.cfg.attn_cfg_self.dtype,
            name="ffn2",
        )(t, deterministic)
        return latents + t


class DecoderBlock(nn.Module):
    attn_cfg: AttnConfig
    rope_cfg: Optional[RoPEConfig]
    ff_hidden: int
    dropout: float

    @nn.compact
    def __call__(self, tokens: Array, token_pos: Array, deterministic: bool) -> Array:
        x = RMSNorm(dtype=self.attn_cfg.dtype)(tokens)
        x = MultiHeadAttention(self.attn_cfg, self.rope_cfg, name="self_attn")(
            x_q=x,
            x_kv=None,
            q_pos=token_pos,
            k_pos=token_pos,
            deterministic=deterministic,
        )
        tokens = tokens + x

        y = RMSNorm(dtype=self.attn_cfg.dtype)(tokens)
        y = SwiGLUFFN(
            self.ff_hidden,
            tokens.shape[-1],
            dropout_rate=self.dropout,
            dtype=self.attn_cfg.dtype,
            name="ffn",
        )(y, deterministic)
        return tokens + y
