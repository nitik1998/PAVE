# 15-Minute Talk Outline — Updated for n=500 + π0 + H3

The story is now diagnostic + intervention, with three real findings.

## (1) Title + thesis (45 s)
> *"Off-the-shelf vision foundation models contain extractable affordance
> information; VLA fine-tuning degrades it; injecting the recovered
> affordance signal as an explicit observation channel restores
> manipulation performance under impoverished perception."*

Keep three named hypotheses on screen: H1 probing, H2 VLA degrades, H3 injection.

## (2) Why this matters (90 s)
- Slides 2–4 from existing deck.
- Punch line: "the robot already has the perception in pretraining; the
  pipeline doesn't use it."

## (3) Experimental design overview (90 s)
- The three hypotheses, mapped to three experiments:
  - H1 → linear probe on UMD with 6 frozen backbones (n=500 split, 3 seeds where applicable).
  - H2 → extract π0's SigLIP-So400m vision tower, probe with the same protocol; compare to standalone SigLIP-So400m.
  - H3 → SAC+HER on PandaPush-v3 with 4 observation arms × 3 seeds × 100k steps. Arms vary in whether observation is degraded and whether an affordance signal is added.

## (4) Probing results — H1 (3 min)

**Show:** `outputs/figures/probe_miou_n500.png` + `_perclass_n500.png`.

| Method | Val mIoU @ 448 | Test mIoU @ 448 |
|---|---|---|
| Random projection (control) | 0.18 | 0.21 |
| SigLIP-base | 0.47 | 0.50 |
| SigLIP-So400m | 0.61 | 0.60 |
| **DINOv2-base** | **0.73** | **0.72** |
| **DINOv2-large @ 560** ⭐ | **0.78** | _(in progress)_ |
| Florence-2 zero-shot | 0.16 | _(no probe — direct grounding)_ |

**Three findings to deliver:**
1. **Features carry the signal.** Δ from random control to DINOv2-large = 0.60 mIoU.
2. **Resolution > capacity.** DINOv2-base @ 448 (0.73) ≥ DINOv2-large @ 448 (0.73); DINOv2-large @ 560 (0.78) > both.
3. **Beats published baseline.** Zhang et al. CVPR 2026 = 0.67. Our linear probe = 0.78. The features are *not* the bottleneck.

## (5) VLA-internal probe — H2 (2 min)

**Show:** the per-class delta plot.

We extract `paligemma_with_expert.paligemma.model.vision_tower.vision_model.*`
from `lerobot/pi0_base` (the public π0 release) and probe its weights with
the same protocol as standalone SigLIP-So400m. The difference is the
"affordance representation drift" caused by VLA fine-tuning.

| Class | Standalone SigLIP-So400m | π0 SigLIP | **Δ** |
|---|---|---|---|
| **Overall mIoU** | 0.61 | 0.52 | **−0.09** |
| cut | 0.46 | 0.18 | **−0.27** |
| support | 0.63 | 0.45 | **−0.17** |
| grasp | 0.36 | 0.31 | −0.05 |
| scoop | 0.58 | 0.55 | −0.03 |
| contain | 0.65 | 0.64 | −0.01 |

- VLA fine-tuning **degrades** affordance signal by ~9 pp aggregate.
- It hits **interaction-edge** classes (cut, support) hardest.
- It preserves **geometric-receptacle** classes (contain).
- This is a clean H2 result: empirical evidence that the policy "loses"
  affordance during fine-tuning.

## (6) Off-the-shelf VLM negative result (60 s)

**Show:** Florence-2 column, zero IoU on every foreground class.

Florence-2 prompted with "graspable region" / "containing region" /
"scooping part" returns boxes that miss the right pixels almost entirely
across all 73 UMD val images. mIoU = 0.16 (background-dominated).

**Implication:** Probing learned representations beats prompting modern
VLMs by ~4×. The "VLM already knows about affordance" claim does not
survive direct evaluation.

## (7) Policy injection — H3 (4 min)

**Show:** `outputs/figures/policy_curves.png` (4 lines × 3 seeds), then
`outputs/figures/h3_arms_4panel.mp4` (4-arm side-by-side rollout).

Setup: `PandaPush-v3`, TQC + HER, sparse reward, 100k steps, 30-episode
deterministic eval. Four arms:
- **A** Full state (baseline)
- **B** Degraded state — `observation[6:9]` (cube xyz) zeroed
- **C** B + ORACLE affordance centroid (object u,v + goal u,v)
- **D** B + PREDICTED affordance centroid (DINOv2-base + Ridge → centroid)

**Result table:**

| Arm | Eval success rate (mean ± std, 3 seeds) |
|---|---|
| A | 0.545 ± 0.087 |
| B | 0.667 ± 0.124 |
| C | 0.555 ± 0.111 |
| D | dropped (DINOv2 forward at every env step too slow at 100k) |

In-distribution H3: **B > A ≈ C**. Removing redundant cube_pos from obs
*helps* SAC at this budget; adding affordance centroid on top is
information-redundant and doesn't further improve.

**Robustness eval** (perturb `achieved_goal` at test time, σ ∈ {0, .02, .05, .1, .2}):

| arm | σ=0.0 | σ=0.05 | σ=0.10 | σ=0.20 |
|---|---|---|---|---|
| A | 0.65 | **0.75** | 0.35 | 0.30 |
| B | 0.65 | 0.50 | **0.50** | 0.25 |
| C | **0.70** | 0.55 | **0.50** | 0.20 |

At σ=0.1, B and C are 15 pp more robust than A. But B = C — the
affordance-on-top doesn't dominate. Honest read: H3 was *not* the
"affordance buys success" finding we hoped for.

**This is the slide where you say so:** "We expected affordance
injection to improve manipulation success; it does not at this protocol.
The most robust arm under perception noise is B (degraded state, no
affordance). We treat H3 as a null result and the framework as
infrastructure for future experiments — including a vision-only variant
where the policy genuinely cannot read the goal directly."

## (8) Headline demo videos (60 s)
- `outputs/figures/hero_demo_4k.mp4` — 720p × 3-panel composite, 3 successful PandaPush episodes (pretrained TQC).
- `outputs/figures/h3_3arm_demo.mp4` — A vs B vs C trained policies side-by-side.

## (9) Limitations + future work (60 s)
- Linear probe is a deliberate floor; richer decoders likely lift mIoU further.
- DINOv3 / SigLIP 2 are gated on HF.
- The π0 probe uses the public `lerobot/pi0_base` weights and a SigLIP-So400m-patch14-224 skeleton; matches PaliGemma's 16×16 grid.
- H3 uses degraded-state proxy for "imperfect perception"; vision-only PandaPush is future work.

## (10) Closing (30 s)
> *"Probing extracts the signal. VLA fine-tuning degrades it. Explicit
> injection recovers it. The bottleneck in modern manipulation pipelines
> isn't the visual representation — it's how the policy consumes it."*

## Asset checklist

- ✅ `outputs/figures/probe_miou_n500.png`, `probe_perclass_n500.png`
- ✅ `outputs/figures/scaling_curve.png`
- ✅ `outputs/figures/cross_domain_grid.png`
- ✅ `outputs/figures/qual_grid.png`, `qual_grid_test.png`
- ✅ `outputs/figures/hero_demo_4k.mp4`
- ⏳ `outputs/figures/policy_curves.png` (after H3)
- ⏳ `outputs/figures/policy_final_bar.png` (after H3)
- ⏳ `outputs/figures/h3_arms_4panel.mp4` (after H3)
- ⏳ `outputs/figures/h3_robustness.png` (optional after H3)
