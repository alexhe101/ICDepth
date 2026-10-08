import torch

from icdepth.configs.model_config import model_loader_configs
from icdepth.models.utils import hash_state_dict_keys, init_weights_on_device
from icdepth.models.wan_video_dit import WanModel

WAN_1_3B = dict(patch_size=[1, 2, 2], in_dim=16, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12, num_layers=30, eps=1e-6)
TINY = dict(patch_size=[1, 2, 2], in_dim=16, dim=48, ffn_dim=96, freq_dim=32, text_dim=32, out_dim=16, num_heads=2, num_layers=2, eps=1e-6)


def random_inputs(frames=3, height=8, width=8):
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 16, frames, height, width, generator=generator)
    reference = torch.randn(1, 16, frames, height, width, generator=generator)
    context = torch.randn(1, 5, TINY["text_dim"], generator=generator)
    dino = torch.randn(1, frames * (height // 2) * (width // 2), 1024, generator=generator)
    return latents, reference, context, dino


def test_checkpoint_layouts_are_registered():
    registered = {config[1] for config in model_loader_configs}
    with init_weights_on_device():
        base = WanModel(**WAN_1_3B)
        icdepth = WanModel(**WAN_1_3B, resolution_cond=True, dino_fea=True)
    assert hash_state_dict_keys(base.state_dict()) in registered
    assert hash_state_dict_keys(icdepth.state_dict()) in registered


def test_new_layers_start_as_identity():
    torch.manual_seed(0)
    base = WanModel(**TINY).eval()
    model = WanModel(**TINY, resolution_cond=True, dino_fea=True).eval()
    model.load_state_dict(base.state_dict(), strict=False)
    model.initialize_missing_modules(set(model.state_dict()) - set(base.state_dict()))
    latents, reference, context, dino = random_inputs()
    with torch.no_grad():
        expected = base(latents, torch.tensor([500.0]), context, reference_latents=reference)
        actual = model(latents, torch.tensor([500.0]), context, reference_latents=reference, dino_feature=dino)
    assert torch.equal(actual, expected)


def test_kv_cache_matches_recomputation():
    torch.manual_seed(0)
    model = WanModel(**TINY, resolution_cond=True, dino_fea=True).eval()
    latents, reference, context, dino = random_inputs()
    kv_cache = [{} for _ in model.blocks]
    with torch.no_grad():
        model(latents, torch.tensor([900.0]), context, reference_latents=reference, dino_feature=dino, kv_cache=kv_cache)
        cached = model(2 * latents, torch.tensor([400.0]), context, reference_latents=reference, dino_feature=dino, kv_cache=kv_cache)
        full = model(2 * latents, torch.tensor([400.0]), context, reference_latents=reference, dino_feature=dino)
    assert torch.allclose(cached, full, atol=1e-5)
