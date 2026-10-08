from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE


# (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
model_loader_configs = [
    # Wan2.1-T2V-1.3B layout, also used by the in-context (ICC) base checkpoint.
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    # ICDepth: Wan2.1-1.3B with the DINOv2 and resolution modulation layers.
    (None, "2d5dfcd0d3de702ab722ca82a0486e80", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
]
