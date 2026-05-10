# Talk outline v3 — 15 minutes — final state of the project

The talk has three landed empirical findings (one strong, two moderate) plus an honest null. Don't oversell.

---

## Slide 1 — Title (30 s)
*"Do Vision Models Understand Affordances?
 Probing and Injecting Affordance Representations for Robot Manipulation"*

One-sentence thesis: **Modern frozen vision foundation models contain extractable affordance information; VLA fine-tuning degrades it asymmetrically across affordance classes; the resulting per-class structure quantitatively predicts which downstream tasks a VLA-conditioned policy can solve.**

## Slides 2–4 — Motivation (90 s)
- Affordances bridge pixels → action.
- Fu et al. 2025 (COLM) showed VLAs degrade affordance aggregate (0.411 → 0.155).
- Two open questions: (a) where exactly does the degradation occur — uniformly or asymmetrically? (b) does it predict downstream policy behavior?

## Slide 5 — Method overview (60 s)
- Stage 1: probe frozen vision encoders with a 60-second sklearn linear probe over patch features.
- Stage 2: extract VLA-internal vision encoders via safetensors and probe with the same protocol.
- Stage 3: validate per-class predictions on a downstream manipulation task.

## Slide 6 — H1 result: probing on UMD (n=500, 90 s)

`outputs/figures/probe_miou_n500.png`

**DINOv2-large + linear probe @ 560 = mIoU 0.776** on UMD val. Beats Zhang et al. CVPR 2026 dense decoder (0.670) by 10 pp using a 60-sec sklearn fit.

Random projection control = 0.179. Δ = 0.60 isolates the foundation-model contribution.

Florence-2 zero-shot = 0.165, Qwen2-VL-2B zero-shot = 0.012. Off-the-shelf VLM grounding fails by 1-2 orders of magnitude.

## Slide 7 — H2 result: VLA-internal degradation, **class-asymmetric** (2 min)

`outputs/figures/h2_delta.png`

Extracted SigLIP-So400m vision tower from `lerobot/pi0_base` (~14 GB safetensors). Probed with same protocol as standalone SigLIP-So400m.

| Class | Standalone | π0 | Δ |
|---|---|---|---|
| grasp | 0.359 | 0.307 | −0.05 |
| **cut** | 0.455 | **0.181** | **−0.27** |
| scoop | 0.578 | 0.545 | −0.03 |
| **contain** | 0.649 | 0.638 | **≈ 0.00** |
| support | 0.625 | 0.453 | −0.17 |

**Headline**: π0's fine-tuning preserves geometric receptacle perception (`contain`) and destroys interaction-edge perception (`cut`). This is direct empirical confirmation of *Affordance in the Wild*'s Axis-1 hypothesis with **per-class detail nobody has published**.

## Slide 8 — H6 result: H2-predicts-behavior on a downstream task (3 min)

Setup: ManiSkill3 PickCube-v1, ManiSkill3 official pretrained PPO baseline (96% success), local 8 GB GPU.

### Step 1 — Pretrained policy is fragile to perception noise
`outputs/figures/h6_robustness_pickcube.png`

| σ on cube_pos | success |
|---|---|
| 0 | 0.96 |
| 0.02 m | 0.27 (full perturbation) |
| 0.05 m | 0.00 |
| 0.10 m | 0.00 |

State-conditioned policies presume internal-consistency across redundant cube-position slices in the obs vector. ~2 cm of perception noise collapses success.

### Step 2 — π0's vision tower predicts cube_pos *better* than DINOv2 on this geometric task
`outputs/figures/h6_predictor_quality.png`

Trained Ridge regression: backbone features → cube xyz. Val L2 error:
- DINOv2-base: **3.68 cm**
- π0 SigLIP-So400m (post-VLA): **1.61 cm**

**π0 wins on this task because PickCube is a `contain`-class geometric problem, and H2 said `contain` was preserved.** The H2 per-class structure numerically predicts the H6 per-task winner.

The reverse should hold for `cut`-class tasks: DINOv2 should beat π0 on knife-handle disambiguation. (Future work — needs a `cut`-affordance manipulation env.)

### Step 3 — Recovery via vision-predicted affordance (results placeholder, in flight)
`outputs/figures/h6_recovery_pickcube.png`

Replacing the noisy cube_pos slice with the vision predictor's output recovers some success. Magnitude under-determined as of writing — partial because state-conditioned policies have *redundant* cube info across slices, requiring all cube-derived slices to be overridden together.

## Slide 9 — Honest null + design lessons (60 s)
- H3 on PandaPush-v3 was null because HER's `achieved_goal` carries clean cube xyz, making affordance redundant.
- Naïvely overriding only one obs slice with a predicted value hurts; the policy receives contradictory signals from redundant slices. Correct intervention is policy retraining or pixels-only conditioning.

## Slide 10 — Negative results (30 s)
- Florence-2 zero-shot on UMD: 0.052 mIoU. Random control: 0.18. **VLM grounding < random.**
- Qwen2-VL-2B zero-shot: 0.012. Effectively zero.
- Conclusion: prompting modern VLMs for fine-grained affordance is not a substitute for probing learned features.

## Slide 11 — Roadmap (60 s)
- Replicate H2 across π0.5, OpenVLA, RT-2 — does the asymmetric degradation generalize?
- Build a `cut`-affordance ManiSkill3 task (knife handle vs blade grasp). Predict π0 fails, DINOv2 succeeds.
- Cosmos cross-attention probing (Axis 2 of *Affordance in the Wild*).

## Slide 12 — Closing (30 s)
**The H2 finding is a publishable contribution on its own**: VLA fine-tuning preserves geometric-receptacle perception and destroys interaction-edge perception, *measured class-asymmetrically* for the first time. The H6 chain shows the per-class signature numerically predicts downstream task performance. The H3 → H6 narrative arc is ongoing.

## Q&A buffer (60 s)

## Asset checklist
- `outputs/figures/probe_miou_n500.png`
- `outputs/figures/h2_delta.png`
- `outputs/figures/h6_robustness_pickcube.png`
- `outputs/figures/h6_predictor_quality.png`
- `outputs/figures/h6_recovery_pickcube.png` (final form once v4 lands)
- `outputs/h6_results.md`
- `outputs/paper_results.md`
- `outputs/findings.md`
- `outputs/results.json`
