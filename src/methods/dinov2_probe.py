"""DINOv2 baseline — same protocol as DINOv3, different backbone."""

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> DinoLinearProbe:
    cfg = ProbeConfig(
        hf_id="facebook/dinov2-base",
        fallback_hf_id=None,
        image_size=448,
        patch_size=14,
        feature_dim=768,
        device=device,
    )
    p = DinoLinearProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
    p.name = "dinov2"
    return p
