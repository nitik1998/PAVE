"""DINOv2-large variant — same protocol, bigger backbone.

`facebook/dinov2-large` is 304M params (vs 86M for base). Open-weight, no auth
required. Patch=14, image config defaults to 224 but supports interpolation.
"""

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> DinoLinearProbe:
    cfg = ProbeConfig(
        hf_id="facebook/dinov2-large",
        fallback_hf_id=None,
        image_size=448,
        patch_size=14,
        feature_dim=1024,
        device=device,
    )
    p = DinoLinearProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
    p.name = "dinov2_large"
    return p
