import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .utils import hash_state_dict_keys

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    """SAND-Attention over the in-context sequence [noisy depth tokens; clean RGB tokens].

    Noisy depth tokens attend to both halves, while RGB tokens only attend to
    themselves, so noise never flows into the condition. Because the RGB half
    does not depend on the noisy half, its keys/values can be cached in
    `kv_cache` after the first denoising step; later calls then pass only the
    noisy tokens.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, kv_cache=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if kv_cache is not None and "k_ref" in kv_cache:
            x = self.attn(
                q,
                torch.cat([k, kv_cache["k_ref"]], dim=1),
                torch.cat([v, kv_cache["v_ref"]], dim=1),
            )
        else:
            split = x.shape[1] // 2
            q_noisy, q_ref = q[:, :split], q[:, split:]
            k_noisy, k_ref = k[:, :split], k[:, split:]
            v_noisy, v_ref = v[:, :split], v[:, split:]
            if kv_cache is not None:
                kv_cache["k_ref"] = k_ref
                kv_cache["v_ref"] = v_ref
            y_noisy = self.attn(
                q_noisy,
                torch.cat([k_noisy, k_ref], dim=1),
                torch.cat([v_noisy, v_ref], dim=1),
            )
            y_ref = self.attn(q_ref, k_ref, v_ref)
            x = torch.cat([y_noisy, y_ref], dim=1)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, kv_cache=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, kv_cache))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class DinoLayer(nn.Module):
    """SRFM semantic branch: modulates the noisy depth tokens with DINOv2 features."""

    def __init__(self, dit_channels, dino_channels):
        super().__init__()
        self.dit_channels = dit_channels
        self.dino_channels = dino_channels
        self.modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dino_channels, dit_channels * 2),
        )

    def zero_init(self):
        nn.init.zeros_(self.modulation_mlp[-1].weight)
        nn.init.zeros_(self.modulation_mlp[-1].bias)

    def forward(self, x, dino_feature, n_noisy=None):
        split = x.shape[1] // 2 if n_noisy is None else n_noisy
        noisy, ref = x[:, :split], x[:, split:]
        scale, shift = self.modulation_mlp(dino_feature).chunk(2, dim=2)
        return torch.cat([noisy * (1 + scale) + shift, ref], dim=1)


class ResolutionEmbedding(nn.Module):
    """Sinusoidal embedding of the latent height and width, followed by an MLP."""

    def __init__(self, freq_dim=256, dim=1536):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def reset_parameters(self):
        for module in self.linear:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, h, w):
        h = h / 256
        w = w / 256
        h_emb = sinusoidal_embedding_1d(self.freq_dim // 2, h)
        w_emb = sinusoidal_embedding_1d(self.freq_dim // 2, w)
        emb = torch.cat([h_emb, w_emb], dim=1)
        return self.linear(emb)


class ResolutionLayer(nn.Module):
    """SRFM resolution branch: modulates all tokens with the resolution embedding."""

    def __init__(self, dit_channels):
        super().__init__()
        self.dit_channels = dit_channels
        self.modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * dit_channels, 2 * dit_channels),
        )

    def zero_init(self):
        nn.init.zeros_(self.modulation_mlp[-1].weight)
        nn.init.zeros_(self.modulation_mlp[-1].bias)

    def forward(self, x, emb):
        scale, shift = self.modulation_mlp(emb).chunk(2, dim=1)
        return x * (1 + scale) + shift


class WanModel(torch.nn.Module):
    """Wan2.1 DiT adapted for video depth estimation with in-context conditioning.

    The depth latents and the RGB latents are concatenated along time and share
    RoPE positions; RGB tokens get a zero timestep. With `dino_fea` and
    `resolution_cond` enabled, every block is followed by the two SRFM
    modulation layers.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        resolution_cond: bool = False,
        dino_fea: bool = False,
        dino_dim: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.resolution_cond = resolution_cond
        self.dino_fea = dino_fea
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        if self.resolution_cond:
            self.resolution_embedding = ResolutionEmbedding(freq_dim=freq_dim, dim=dim)
            self.resolution_layers = nn.ModuleList([ResolutionLayer(dim) for _ in range(num_layers)])
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        if self.dino_fea:
            self.fusion_blocks = nn.ModuleList([DinoLayer(dim, dino_dim) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

    def initialize_missing_modules(self, missing_keys):
        """Initialize ICDepth layers that a Wan2.1 / ICC base checkpoint does not contain.

        The modulation layers start at zero, so the initialized model behaves
        exactly like the base checkpoint.
        """
        missing_keys = set(missing_keys)
        initialized = set()

        def take(prefix):
            keys = {key for key in missing_keys if key.startswith(prefix)}
            initialized.update(keys)
            return len(keys) > 0

        if self.dino_fea:
            for block_id, block in enumerate(self.fusion_blocks):
                if take(f"fusion_blocks.{block_id}."):
                    block.zero_init()
        if self.resolution_cond:
            if take("resolution_embedding."):
                self.resolution_embedding.reset_parameters()
            for block_id, block in enumerate(self.resolution_layers):
                if take(f"resolution_layers.{block_id}."):
                    block.zero_init()
        remaining = missing_keys - initialized
        if remaining:
            raise ValueError(f"The DiT checkpoint is missing {len(remaining)} parameters, e.g. {sorted(remaining)[:5]}.")

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        reference_latents: torch.Tensor = None,
        dino_feature: torch.Tensor = None,
        kv_cache: list = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        """Predict the flow-matching velocity of the depth latents.

        Args:
            latents: Noisy depth latents, (B, C, F, H, W).
            timestep: Diffusion timestep, shape (1,).
            context: Text embeddings, (B, L, text_dim).
            reference_latents: Clean RGB latents with the same shape as `latents`.
                Ignored once `kv_cache` holds the reference keys/values.
            dino_feature: DINOv2 features aligned to the depth tokens, (B, N, dino_dim).
            kv_cache: Optional list with one dict per block for caching the
                reference keys/values across denoising steps.
        """
        use_reference = reference_latents is not None and not (kv_cache is not None and len(kv_cache[0]) > 0)

        tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
        timesteps = [torch.ones((latents.shape[2], tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep]
        if use_reference:
            timesteps.append(torch.zeros((reference_latents.shape[2], tokens_per_frame), dtype=latents.dtype, device=latents.device))
        timestep = torch.concat(timesteps, dim=0).flatten()
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).unsqueeze(0))
        t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)
        if self.resolution_cond:
            h, w = latents.shape[3], latents.shape[4]
            t_res = self.resolution_embedding(
                torch.tensor([h], dtype=t.dtype, device=t.device),
                torch.tensor([w], dtype=t.dtype, device=t.device),
            )

        x = latents
        if use_reference:
            if reference_latents.shape[0] != x.shape[0]:
                reference_latents = torch.concat([reference_latents] * x.shape[0], dim=0)
            x = torch.cat([x, reference_latents], dim=2)
        x, (f, h, w) = self.patchify(x)
        n_noisy = None if use_reference else f * h * w

        # RoPE alignment: the depth and RGB halves share the same (t, h, w) positions.
        if use_reference:
            freqs_t = torch.cat([self.freqs[0][:f // 2]] * 2, dim=0)
        else:
            freqs_t = self.freqs[0][:f]
        freqs = torch.cat([
            freqs_t.view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def run(module, *inputs):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    return torch.utils.checkpoint.checkpoint(module, *inputs, use_reentrant=False)
            if use_gradient_checkpointing:
                return torch.utils.checkpoint.checkpoint(module, *inputs, use_reentrant=False)
            return module(*inputs)

        for block_id, block in enumerate(self.blocks):
            block_kv = kv_cache[block_id] if kv_cache is not None else None
            x = run(block, x, context, t_mod, freqs, block_kv)
            if self.dino_fea:
                x = run(self.fusion_blocks[block_id], x, dino_feature, n_noisy)
            if self.resolution_cond:
                x = run(self.resolution_layers[block_id], x, t_res)

        x = self.head(x, t)
        if use_reference:
            x = x[:, :x.shape[1] // 2]
            f = f // 2
        return self.unpatchify(x, (f, h, w))

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) in (
            "9269f8db9040a9d860eaca435be61814",  # Wan2.1-T2V-1.3B / ICC base
            "2d5dfcd0d3de702ab722ca82a0486e80",  # ICDepth
        ):
            config = {
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "resolution_cond": True,
                "dino_fea": True,
            }
        else:
            raise ValueError("Unsupported Wan DiT checkpoint.")
        return state_dict, config
