# simplified_perceiver_jax.py
# - Perceiver-IO encoder/decoder (KISS version)
# - GQA/MQA, QK-norm, simple scaled dot-product attention
# - Fourier encodings for inputs
# - Optional RoPE on decoder queries
# - JAX/Flax best practices

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat

Array = jax.Array


def fourier_encode(x: Array, num_encodings: int = 4) -> Array:
    """Sin/Cos Fourier features + raw, last-dim is position(s)."""
    x = jnp.expand_dims(x, -1)  # (..., C) -> (..., C, 1)
    orig_x = x
    scales = (2.0 ** jnp.arange(num_encodings)).astype(x.dtype)  # (E,)
    x = x / scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)  # (..., C, 2E)
    x = jnp.concatenate([x, orig_x], axis=-1)               # (..., C, 2E+1)
    return x


class RMSNorm(nn.Module):
    eps: float = 1e-6
    learnable_scale: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.eps)
        y = (x_f32 / rms).astype(dtype)
        if self.learnable_scale:
            scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), dtype=dtype)
            y = y * scale
        return y


def qk_norm(q: Array, k: Array, eps: float = 1e-6) -> Tuple[Array, Array]:
    """Per-head unit-variance normalization for Q, K."""
    def _norm(a):
        a32 = a.astype(jnp.float32)
        var = jnp.mean(jnp.square(a32), axis=-1, keepdims=True)
        return (a32 * jax.lax.rsqrt(var + eps)).astype(a.dtype)
    return _norm(q), _norm(k)


def swiglu(x: Array, limit: float = 7.0, alpha: float = 1.702) -> Array:
    """SwiGLU with input clamps."""
    x_glu, x_lin = jnp.split(x, 2, axis=-1)
    x_glu = jnp.clip(x_glu, a_min=None, a_max=limit)
    x_lin = jnp.clip(x_lin, a_min=-limit, a_max=limit)
    out_glu = x_glu * jax.nn.sigmoid(alpha * x_glu)
    return out_glu * (x_lin + 1.0)


@dataclass
class RoPEConfig:
    head_dim: int
    base_theta: float = 10000.0
    initial_context: int = 4096
    scaling_factor: float = 1.0
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0


def _rope_concentration_and_inv_freq(cfg: RoPEConfig):
    """YaRN-like concentration/inv_freq."""
    d = cfg.head_dim
    d_half = d // 2
    freq = cfg.base_theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d)
    if cfg.scaling_factor > 1.0:
        concentration = 0.1 * jnp.log(cfg.scaling_factor) + 1.0
        low = d_half * jnp.log(cfg.initial_context / (cfg.ntk_beta * 2 * jnp.pi)) / jnp.log(cfg.base_theta)
        high = d_half * jnp.log(cfg.initial_context / (cfg.ntk_alpha * 2 * jnp.pi)) / jnp.log(cfg.base_theta)
        ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
        mask = 1.0 - jnp.clip(ramp, 0.0, 1.0)
        inv_freq = (1.0 / (cfg.scaling_factor * freq)) * (1.0 - mask) + (1.0 / freq) * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq
    return concentration, inv_freq


def apply_rope(q: Array, k: Array, pos: Array, cfg: RoPEConfig) -> Tuple[Array, Array]:
    """
    Apply RoPE to q and k.
    q: (B, Q, H, D), k: (B, K, Hkv, D), pos: (B, Q) integer positions
    """
    assert (q.shape[-1] % 2 == 0) and (k.shape[-1] % 2 == 0), "RoPE requires even head_dim"
    conc, inv_freq = _rope_concentration_and_inv_freq(cfg)
    t_q = pos.astype(jnp.float32)  # (B, Q)
    freqs_q = t_q[..., None] * inv_freq[None, ...]  # (B, Q, D/2)
    cos_q, sin_q = jnp.cos(freqs_q) * conc, jnp.sin(freqs_q) * conc
    t_k = jnp.arange(k.shape[1], dtype=jnp.float32)[None, :].repeat(k.shape[0], axis=0)
    freqs_k = t_k[..., None] * inv_freq[None, ...]
    cos_k, sin_k = jnp.cos(freqs_k) * conc, jnp.sin(freqs_k) * conc

    def _rotate(x, cos, sin):
        x1, x2 = jnp.split(x, 2, axis=-1)
        cos = cos[..., None, :].astype(x.dtype)
        sin = sin[..., None, :].astype(x.dtype)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return jnp.concatenate([o1, o2], axis=-1)

    return _rotate(q, cos_q, sin_q), _rotate(k, cos_k, sin_k)


def repeat_kv(kv: Array, n_rep: int) -> Array:
    """Repeat KV heads for GQA/MQA. kv: (B, Tk, Hkv, D) -> (B, Tk, H, D)"""
    if n_rep == 1:
        return kv
    B, Tk, Hkv, D = kv.shape
    kv = kv[:, :, :, None, :].repeat(n_rep, axis=3)
    return rearrange(kv, "b t hkv r d -> b t (hkv r) d")


def scaled_dot_product_attention(q: Array, k: Array, v: Array,
                                 mask: Optional[Array] = None,
                                 bias: Optional[Array] = None) -> Array:
    """
    Standard scaled dot-product attention.
    q: (B, Tq, H, D), k/v: (B, Tk, H, D)
    mask: broadcastable to (B, H, Tq, Tk), True=keep
    bias: broadcastable to (B, H, Tq, Tk), added to logits
    """
    B, Tq, H, D = q.shape
    Tk = k.shape[1]
    scale = jnp.asarray(1.0 / jnp.sqrt(D), dtype=q.dtype)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", attn, v)
    return out


class SwiGLUFFN(nn.Module):
    """Simple SwiGLU FFN."""
    hidden_size: int
    out_size: int
    dtype: jnp.dtype = jnp.bfloat16
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, deterministic: bool) -> Array:
        h = nn.Dense(self.hidden_size * 2, use_bias=False, dtype=self.dtype,
                     kernel_init=nn.initializers.xavier_uniform())(x)
        h = swiglu(h).astype(self.dtype)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        y = nn.Dense(self.out_size, use_bias=False, dtype=self.dtype,
                     kernel_init=nn.initializers.xavier_uniform())(h)
        return y


@dataclass
class AttnConfig:
    num_heads: int
    head_dim: int
    num_kv_heads: int
    dtype: jnp.dtype = jnp.bfloat16
    use_qk_norm: bool = True


class MultiHeadAttention(nn.Module):
    cfg: AttnConfig

    @nn.compact
    def __call__(self,
                 x_q: Array,
                 x_kv: Optional[Array] = None,
                 mask: Optional[Array] = None,
                 attn_bias: Optional[Array] = None,
                 deterministic: bool = True) -> Array:
        """
        x_q: (B, Tq, C), x_kv: (B, Tk, C) [None -> self-attn]
        returns: (B, Tq, C)
        """
        cfg = self.cfg
        dtype = cfg.dtype
        H, D = cfg.num_heads, cfg.head_dim
        Hkv = cfg.num_kv_heads
        assert H % Hkv == 0, "num_heads must be multiple of num_kv_heads"
        rep = H // Hkv

        C = x_q.shape[-1]
        x_kv = x_q if x_kv is None else x_kv

        # Projections
        q = nn.DenseGeneral((H, D), use_bias=False, dtype=dtype,
                            kernel_init=nn.initializers.xavier_uniform(), name="q")(x_q)
        k = nn.DenseGeneral((Hkv, D), use_bias=False, dtype=dtype, name="k")(x_kv)
        v = nn.DenseGeneral((Hkv, D), use_bias=False, dtype=dtype, name="v")(x_kv)

        # QK norm
        if cfg.use_qk_norm:
            q, k = qk_norm(q, k)

        # Repeat KV heads (GQA/MQA)
        k = repeat_kv(k, rep)
        v = repeat_kv(v, rep)

        # Attention
        out = scaled_dot_product_attention(q, k, v, mask=mask, bias=attn_bias)

        # Merge heads then project back to model dimension
        out = rearrange(out, "b q h d -> b q (h d)")
        y = nn.Dense(C, use_bias=False, dtype=dtype,
                     kernel_init=nn.initializers.xavier_uniform(), name="o")(out)
        return y


@dataclass
class PerceiverBlockConfig:
    attn_cfg_cross: AttnConfig
    attn_cfg_self: AttnConfig
    ff_hidden: int
    ff_dropout: float = 0.0


class PerceiverEncoderBlock(nn.Module):
    """Single Perceiver encoder block: cross-attn + FFN + self-attn + FFN."""
    cfg: PerceiverBlockConfig

    @nn.compact
    def __call__(self, latents: Array, inputs: Array, deterministic: bool) -> Array:
        # Cross-attend (latents query inputs)
        x = RMSNorm()(latents)
        x = MultiHeadAttention(self.cfg.attn_cfg_cross, name="cross_attn")(
            x_q=x, x_kv=inputs, deterministic=deterministic
        )
        latents = latents + x
        
        # FFN
        y = RMSNorm()(latents)
        y = SwiGLUFFN(self.cfg.ff_hidden, latents.shape[-1],
                      dtype=self.cfg.attn_cfg_cross.dtype,
                      dropout_rate=self.cfg.ff_dropout, name="ffn")(y, deterministic)
        latents = latents + y
        
        # Latent self-attention
        z = RMSNorm()(latents)
        z = MultiHeadAttention(self.cfg.attn_cfg_self, name="latent_self_attn")(
            x_q=z, x_kv=None, deterministic=deterministic
        )
        latents = latents + z
        
        # FFN again
        t = RMSNorm()(latents)
        t = SwiGLUFFN(self.cfg.ff_hidden, latents.shape[-1],
                      dtype=self.cfg.attn_cfg_self.dtype,
                      dropout_rate=self.cfg.ff_dropout, name="ffn2")(t, deterministic)
        latents = latents + t
        
        return latents


class PerceiverIO(nn.Module):
    """
    Simplified Perceiver-IO:
      - Encode: inputs -> latents (repeated cross + latent self-attn)
      - Decode: queries -> output tokens via cross-attn into latents
    """
    num_latents: int = 256
    latent_dim: int = 512
    depth: int = 6
    attn_cfg_cross: AttnConfig = None
    attn_cfg_self: AttnConfig = None
    ff_hidden: int = 2048
    ff_dropout: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    # Decoder options
    decoder_dim: int = 512
    decoder_heads: int = 8
    decoder_kv_heads: int = 8
    decoder_head_dim: int = 64
    rope_cfg: Optional[RoPEConfig] = None

    @nn.compact
    def __call__(self,
                 inputs: Array,                 # (B, N, Cin)
                 input_pos: Optional[Array],    # (B, N, P) pos channels
                 queries: Array,                # (B, M, Cq)
                 query_pos_ids: Optional[Array] = None,  # (B, M) for RoPE
                 deterministic: bool = True) -> Array:
        dtype = self.dtype
        B = inputs.shape[0]

        # 1) Input encoding with Fourier features
        if input_pos is not None:
            pos_feat = fourier_encode(input_pos, num_encodings=4)
            pos_feat = rearrange(pos_feat, "... p c -> ... (p c)")
            x = jnp.concatenate([inputs, pos_feat.astype(inputs.dtype)], axis=-1)
        else:
            x = inputs

        # 2) Initialize latents
        latents = self.param("latents", nn.initializers.normal(stddev=0.02),
                             (self.num_latents, self.latent_dim), dtype)
        latents = repeat(latents, "n d -> b n d", b=B).astype(dtype)

        # 3) Stack Perceiver encoder blocks
        block_cfg = PerceiverBlockConfig(
            attn_cfg_cross=self.attn_cfg_cross,
            attn_cfg_self=self.attn_cfg_self,
            ff_hidden=self.ff_hidden,
            ff_dropout=self.ff_dropout,
        )

        for i in range(self.depth):
            latents = PerceiverEncoderBlock(block_cfg, name=f"encoder_block_{i}")(
                latents, x, deterministic
            )

        # 4) Perceiver-IO decoder: queries attend latents
        q = RMSNorm()(queries.astype(dtype))
        dec_cfg = AttnConfig(
            num_heads=self.decoder_heads,
            head_dim=self.decoder_head_dim,
            num_kv_heads=self.decoder_kv_heads,
            dtype=dtype,
            use_qk_norm=True,
        )

        # Optional RoPE on decoder queries
        if self.rope_cfg is not None and query_pos_ids is not None:
            q_proj = nn.DenseGeneral((dec_cfg.num_heads, dec_cfg.head_dim), 
                                    use_bias=False, dtype=dtype, name="dec_q")(q)
            k_proj = nn.DenseGeneral((dec_cfg.num_kv_heads, dec_cfg.head_dim), 
                                    use_bias=False, dtype=dtype, name="dec_k")(latents)
            v_proj = nn.DenseGeneral((dec_cfg.num_kv_heads, dec_cfg.head_dim), 
                                    use_bias=False, dtype=dtype, name="dec_v")(latents)
            rep = dec_cfg.num_heads // dec_cfg.num_kv_heads
            k_proj = repeat_kv(k_proj, rep)
            v_proj = repeat_kv(v_proj, rep)
            q_proj, k_proj = apply_rope(q_proj, k_proj, query_pos_ids, self.rope_cfg)
            y = scaled_dot_product_attention(q_proj, k_proj, v_proj)
            y = nn.DenseGeneral((self.decoder_dim,), use_bias=False, dtype=dtype, 
                               name="dec_o")(y)
        else:
            # Standard decoder cross-attn
            y = MultiHeadAttention(dec_cfg, name="decoder_cross_attn")(
                q, latents, deterministic=deterministic
            )

        return y  # (B, M, decoder_dim)
