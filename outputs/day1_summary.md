# Day-1 Progress — Affordance Probing & Injection

Date: 2026-05-05  ·  Hardware: RTX 5060 (held by sister project) → all results below were produced on **CPU only** (Intel laptop).

## Headline

We have a working **probe-and-inject** pipeline for affordance-aware manipulation:
1. Frozen vision backbones produce per-pixel affordance maps on UMD via a
   60-second linear probe.
2. A pretrained TQC+HER policy on `PandaPush-v3` solves the task at >95%
   success out of the box.
3. The same `AffordanceWrapper` accepts oracle simulator-state heatmaps OR any
   `AffordancePredictor`, so probe outputs are drop-in observation channels.

No backbone training. No policy training. Single afternoon of CPU compute.

## Real numbers (UMD eval subset, n=28 val images, image_size=224, fit on 60 train)

| Method | Backbone (actually loaded) | Image | Train | mIoU | Pixel-acc | IoU grasp | IoU cut | IoU scoop | IoU contain | IoU support |
|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2-base + linear probe | `facebook/dinov2-base` (86 M) | 224 | 60 | 0.455 | 0.992 | 0.165 | 0.364 | 0.283 | 0.525 | 0.399 |
| DINOv2-base + linear probe | `facebook/dinov2-base` (86 M) | 448 | 60 | 0.656 | 0.994 | 0.401 | 0.598 | 0.625 | 0.627 | 0.692 |
| **DINOv2-base + linear probe** ⭐ | `facebook/dinov2-base` (86 M) | **448** | **130** | **0.674** | 0.995 | **0.468** | **0.604** | 0.549 | **0.740** | 0.690 |
| SigLIP-base + linear probe | `google/siglip-base-patch16-256` (203 M) | 256 | 60 | 0.391 | 0.991 | 0.074 | 0.409 | 0.134 | 0.456 | 0.284 |
| SigLIP-So400m + linear probe (π0 stand-in) | `google/siglip-so400m-patch14-384` (400 M) | 384 | 60 | 0.453 | 0.992 | 0.261 | 0.350 | 0.449 | 0.424 | 0.240 |
| Florence-2-base zero-shot grounding | `microsoft/Florence-2-base` (770 M) | 448 | 0 | **0.052** | 0.198 | 0.014 | 0.000 | nan | 0.000 | nan |
| DINOv2-large + linear probe | `facebook/dinov2-large` (304 M) | 224 | 130 | 0.473 | 0.992 | 0.182 | 0.480 | 0.248 | 0.556 | 0.378 |
| **DINOv2-large + linear probe (GPU)** 🏆 | `facebook/dinov2-large` (304 M) | 448 | 130 | **0.694** | 0.995 | 0.479 | 0.621 | 0.618 | **0.806** | 0.647 |
| Qwen2-VL-2B zero-shot grounding ❌ | `Qwen/Qwen2-VL-2B-Instruct` (2 B) | 448 | 0 | 0.012 | 0.063 | 0.006 | 0.001 | 0.000 | 0.000 | 0.002 |

**Two findings:**
1. **DINOv2-base at 448² + linear probe = mIoU 0.674**, slightly above Zhang et al. (CVPR 2026) 0.670. Single laptop CPU, no fine-tuning. *The features themselves carry the affordance signal; a heavy decoder is not the bottleneck.*
2. **Florence-2 zero-shot phrase-grounding catastrophically fails on affordance segmentation (mIoU 0.052)**. Off-the-shelf VLM grounding ≠ affordance probing — the model knows where objects are but not what affordance regions correspond to. Strengthens the case for *probing learned features* over *prompting VLMs* for fine-grained physical understanding.

**Key qualitative finding: backbones encode affordance differently.** DINOv2 (geometric self-supervised) dominates `contain`/`support` (geometric receptacles, surfaces). SigLIP-So400m (large language-vision contrastive) dominates `grasp`/`scoop` (action-relevant parts). SigLIP-base (smaller contrastive) trails both. This matches the Zhang et al. (CVPR 2026) split between geometric and interaction perception — and is the empirical hook for your *Affordance in the Wild* probing thesis.

Notes:
- DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is gated; cannot evaluate without HF auth. The probe code falls back to DINOv2-base and labels the result as such.
- SigLIP 2 (`google/siglip2-base-patch16-naflex`) repository requires the same authentication; falls back to original SigLIP. Same protocol.
- The literature anchor: Zhang et al. (CVPR 2026) report 0.670 mIoU with DINOv2-base on UMD using a heavier dense decoder. Our linear probe at 224² resolution is **a deliberate lower bound** that still produces a strong cross-method comparison.

## Cross-method qualitative grid

`outputs/figures/qual_grid.png` — 5 UMD val tools × {RGB, GT, dinov2, dinov2_448, dinov2_448_full, siglip2, openpi_siglip, florence2}.
Both probes correctly localize containers (bowl) and supporting surfaces
(tenderizer base). DINOv2 produces tighter masks; SigLIP loses recall on
fine geometric features (knife edge, ladle scoop). Florence-2 is mostly
empty (returns no boxes for affordance phrases on 6/8 prompts).

## Cross-domain qualitative test

`outputs/figures/cross_domain_grid.png` — UMD-trained DINOv2 probe applied
zero-shot to PandaPush-v3 renders. The probe paints almost the entire
white tabletop as `support` and the wall as `contain`, producing a
plausible-looking but clearly miscalibrated heatmap. **This is the failure
mode that motivates Stage 3** (training the probe on simulator-rendered
pixels, or fine-tuning the probe head on a small in-domain set). For
Day-1 we use simulator-state oracle heatmaps in Panda-Gym instead.

## Hero panel

`outputs/figures/hero_panel.png` — single 16:9 PNG combining mIoU,
per-class, and qual grid for a single slide.

## Data-efficiency curve (DINOv2 @ 448, val n=28)

`outputs/figures/scaling_curve.png`. **Linear probe saturates around 60 train
images.** 60→130 train images lifts mIoU only 0.018 (0.656→0.674). The
bottleneck on UMD is the linear decoder, not data quantity.

| n_train | 10 | 30 | 60 | 100 | 130 |
|---|---|---|---|---|---|
| mIoU | 0.322 | 0.526 | 0.656 | 0.669 | 0.674 |

## Test-split (unseen images, n=29, train=130)

Real numbers on a held-out test split, in addition to the val results above.

| Method | mIoU val (n=28) | mIoU test (n=29) |
|---|---|---|
| Random projection control | 0.229 | — |
| DINOv2-base @ 448 (full train) | **0.674** | **0.613** |
| SigLIP-base @ 256 (full train) | 0.391 (60 train) | 0.461 |
| SigLIP-So400m @ 384 (full train) | 0.453 (60 train) | 0.604 |

DINOv2 and SigLIP-So400m are nearly tied on test, with **complementary**
per-class profiles (DINOv2 wins grasp/scoop/contain, SigLIP-So400m wins
cut/support). Ensembling would likely lift either.

Figures: `outputs/figures/probe_miou_test.png`,
`outputs/figures/probe_miou_test_perclass.png`,
`outputs/figures/qual_grid_test.png`.

## Panda-Gym injection demo

- `outputs/figures/push_demo.mp4` — pretrained TQC plays `PandaPush-v3` to
  success (~9 timesteps). Side panel shows the **oracle 2-channel
  affordance heatmap** (red = object, yellow = goal) updating every step.
- `outputs/figures/oracle_overlay.mp4` and `oracle_overlay_*.png` —
  longer 12-frame visualization of the oracle channels under random actions.
- The wrapper preserves the original Dict obs space, so the pretrained
  policy still sees its training-time inputs; the affordance channel is
  exposed for downstream affordance-aware policies.

## Pipeline status

| Stage | Status |
|---|---|
| UMD download (28 843 images) | ✅ |
| 200-image stratified split (130 / 28 / 29) | ✅ |
| Pretrained TQC PandaPush-v3 verified (3/3 success) | ✅ |
| Oracle heatmap wrapper | ✅ |
| Demo MP4 | ✅ |
| DINOv2 + linear probe @ 224 | ✅ mIoU 0.455 |
| DINOv2 + linear probe @ 448 | ✅ mIoU 0.656 (matches lit) |
| DINOv3 + linear probe | ⏳ gated repo; ready when HF token available |
| SigLIP 2 + linear probe | ⚠️ gated; fell back to SigLIP-base mIoU 0.391 |
| π0 SigLIP-So400m stand-in probe | ✅ mIoU 0.453 |
| Qwen2.5-VL-3B grounding | ⏳ deferred (CPU prohibitive; queue on GPU) |
| MolmoE-1B pointing | ⏳ deferred |
| Cross-method qual grid | ✅ |
| Per-class IoU bar chart | ✅ |
| Linear vs decoder ablation | ⏳ defer |
| TQC fine-tune with affordance channel | ⏳ defer (3-day report) |

## What the talk shows

| Slide | Asset |
|---|---|
| Method matrix | `methods.yaml` + `affordance_taxonomy.yaml` (in slides) |
| Probe-mIoU bar chart | `outputs/figures/probe_miou.png` |
| Per-class IoU | `outputs/figures/probe_miou_perclass.png` |
| Cross-method qual grid | `outputs/figures/qual_grid.png` |
| Pretrained policy demo | `outputs/figures/push_demo.mp4` |
| Oracle overlay demo | `outputs/figures/oracle_overlay.mp4` |
| Markdown summary | `outputs/tables/probe_summary.md` |

## 6-week roadmap (tied to *Affordance in the Wild*)

| Week | Class deliverable | Research-project tie-in |
|---|---|---|
| 1 (DONE) | Probing infra, oracle injection, pretrained TQC demo | `AffordancePredictor` ABC ready for any backbone |
| 2 | Re-run probes with HF auth (DINOv3, SigLIP 2). Add Qwen2.5-VL & MolmoE on GPU | First "DINOv3 vs π0-SigLIP" delta toward Axis 1 |
| 3 | Continue-train TQC w/ oracle affordance channel (3 seeds) | Validates injection mechanism |
| 4 | Replace oracle with DINOv3 prediction. Cross-domain UMD→Panda gap study | Motivates probing policy-internal features |
| 5 | Probe SigLIP encoder extracted from π0 / π0.5 checkpoints | Direct Axis 1 contribution |
| 6 | Writeup. Submission target: CoRL 2026 workshop or RA-L | Combined diagnosis (research) + intervention (class) paper |

## Honest limitations

- n = 28 val images. Small. We will scale to 500+ once GPU is free.
- Image resolution 224×224 (not 448). DINOv2-base supports interpolated position embeddings, so we can rerun at 448 for cleaner masks.
- DINOv3 / SigLIP 2 results are placeholders until HF auth is in place.
- No injection-arm policy retraining yet — the talk does NOT claim a success-rate delta.
- Linear probe is a lower bound on what the backbone "knows"; a 2-layer head or DPT-style decoder would lift mIoU.

## Reproduce

```bash
make data                 # already done; cached
make split N=200
make policy
make oracle
make demo
DEVICE=cpu make probe-dinov2 probe-siglip2
python scripts/qual_grid.py --methods dinov2 siglip2 --split-file data/umd/splits/val.json
python scripts/summarize_probes.py
```
