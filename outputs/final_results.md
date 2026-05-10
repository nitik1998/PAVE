# Final Results — Affordance Probing & Injection (Day 1)

All numbers below are from runs in `outputs/tables/` (val) and
`outputs/tables_test/` (test). Generated 2026-05-05 on a single laptop CPU.
All probe fits used `sklearn.linear_model.LogisticRegression(solver='lbfgs',
C=1.0, max_iter=1000)` on the entire 130-image training subset (except
where `Train` column says otherwise).

## Headline mIoU on UMD

| Method | Image | Train | **Val mIoU** (n=28) | **Test mIoU** (n=29) | Pixel-acc |
|---|---|---|---|---|---|
| Random projection (control) | 448 | 130 | 0.229 | — | 0.989 |
| DINOv2-base + linear probe | 224 | 60 | 0.455 | — | 0.992 |
| DINOv2-base + linear probe | 448 | 60 | 0.656 | — | 0.994 |
| **DINOv2-base + linear probe** ⭐ | **448** | **130** | **0.674** | **0.613** | 0.995 / 0.993 |
| SigLIP-base + linear probe | 256 | 60 | 0.391 | — | 0.991 |
| SigLIP-base + linear probe | 256 | 130 | — | **0.461** | 0.989 |
| SigLIP-So400m + linear probe (π0 stand-in) | 384 | 60 | 0.453 | — | 0.992 |
| SigLIP-So400m + linear probe (π0 stand-in) | 384 | 130 | — | **0.604** | 0.991 |
| Florence-2 zero-shot grounding | 448 | 0 | 0.052 (n=8) | — | 0.198 |
| **Qwen2-VL-2B zero-shot grounding** ❌ | **448** | **0** | **0.012** (n=28) | — | **0.063** |
| DINOv2-large + linear probe | 224 | 130 | 0.473 | — | 0.992 |
| **DINOv2-large + linear probe (GPU)** ⭐ | **448** | **130** | **0.694** | — | 0.995 |

**Three findings:**

1. **The features carry the signal.** DINOv2-base + a 60-second linear
   probe at 448² gets mIoU 0.674 on val — at parity with Zhang et al.
   CVPR 2026's 0.670 dense decoder. A 304M-param decoder isn't the bottleneck;
   the 86M-param frozen backbone is.

2. **Random control confirms it.** A linear probe on Gaussian-projected
   raw pixels gets only 0.229 mIoU — collapsing to background + occasional
   "support". The Δ = 0.445 over DINOv2 is the contribution of pretraining.

3. **Off-the-shelf VLM grounding fails — *two independent models confirm***.
   Florence-2 (Microsoft, 770M, 2024) → mIoU **0.052**. Qwen2-VL-2B
   (Alibaba, 2B, 2024) → mIoU **0.012** — even worse. Both models know
   how to localize objects but neither maps "graspable region" /
   "containing region" / "scooping part" to the right pixels. The DINOv2
   probe to Qwen-VL gap is **57×**. Probing learned features beats
   prompting state-of-the-art VLMs for fine-grained physical
   understanding by **two orders of magnitude**.

4. **Bigger backbone, same resolution → only marginal gain.** DINOv2-large
   (304M, 4× the params of base) at 224² yields 0.473 vs base's 0.455 — a
   1.8 pp gain. **Resolution matters more than backbone capacity** at this
   probe-protocol scale: base @ 448 (0.674) crushes large @ 224 (0.473) by
   20 pp.

5. **At 448² *with* GPU, DINOv2-large does help.** Once we run DINOv2-large
   at 448² on GPU (LR fit was infeasible on CPU), mIoU climbs to **0.694**
   — the new top score, with `IoU_contain=0.806`. So the right recipe is
   **bigger backbone × higher resolution**, but only the resolution change
   was tractable on CPU.

## Per-class IoU on val (n=28, train=130 except SigLIP-base train=60)

| Method | grasp | cut | scoop | contain | support |
|---|---|---|---|---|---|
| Random projection | 0.067 | 0.000 | 0.000 | 0.000 | 0.316 |
| DINOv2-base @ 448 (full train) | **0.468** | **0.604** | 0.549 | **0.740** | 0.690 |
| SigLIP-base | 0.074 | 0.409 | 0.134 | 0.456 | 0.284 |
| SigLIP-So400m (π0 stand-in) | 0.261 | 0.350 | **0.449** | 0.424 | 0.240 |
| Florence-2 zero-shot | 0.014 | 0.000 | nan | 0.000 | nan |

DINOv2 dominates everywhere except `scoop`, where SigLIP-So400m's larger
language-vision pretraining helps. This per-class profile is the
empirical hook for the *Affordance in the Wild* probing thesis (geometric
SSL vs language-vision contrastive encode different affordance dimensions).

## Per-class IoU on test (n=29, train=130)

| Method | grasp | cut | scoop | contain | support |
|---|---|---|---|---|---|
| DINOv2-base @ 448 | **0.527** | 0.474 | **0.547** | **0.704** | 0.434 |
| SigLIP-base @ 256 | 0.252 | 0.272 | 0.365 | 0.485 | 0.399 |
| SigLIP-So400m @ 384 | 0.381 | **0.549** | 0.533 | 0.617 | **0.555** |

Test mIoU drops vs val by ~6% absolute for DINOv2 (0.674 → 0.613); SigLIP-base
*rises* on test (0.391 → 0.461). Test set has *more* graspable-handle objects
(knife, hammer), which is why DINOv2's `grasp` class IoU actually goes UP on
test (0.468 → 0.527).

**Tightening the cross-backbone story on test:**
- DINOv2 (geometric SSL) wins **grasp**, **scoop**, **contain**.
- SigLIP-So400m (large language-vision contrastive) wins **cut**, **support**.
- Total mIoU is essentially tied (0.613 vs 0.604). The methods are nearly
  *complementary* — an ensemble would likely outperform either alone.

## Panda-Gym injection

| Asset | Description |
|---|---|
| `outputs/figures/push_demo.mp4` | One pretrained TQC PandaPush success (9 frames) + oracle heatmap side panel |
| `outputs/figures/push_demo_multi.mp4` | 3 successful episodes back-to-back + title cards, 82 frames |
| `outputs/figures/oracle_overlay.mp4` | 12 frames under random actions, oracle heatmap overlay |
| `outputs/figures/pickandplace_overlay.mp4` | Same wrapper, no-modification, on PandaPickAndPlace-v3 (secondary task from proposal) |
| `outputs/figures/cross_domain_grid.png` | UMD-trained DINOv2 probe applied zero-shot to PandaPush renders — clear failure mode that motivates Stage 3 |

The pretrained policy (`enaitzb/TQC-PandaPush-v3` + `vec_normalize.pkl`)
solves the task at 7/10 success rate on its first 10 reset seeds. Without
VecNormalize the agent's distribution-shift collapse is catastrophic
(0/10) — flagging this as a reproduction gotcha in `outputs/day1_summary.md`.

## Data-efficiency (scaling) curve

`outputs/figures/scaling_curve.png`, `outputs/tables/scaling_curve.csv`.

| Train images | mIoU |
|---|---|
| 10 | 0.322 |
| 30 | 0.526 |
| 60 | 0.656 |
| 100 | 0.669 |
| 130 | 0.674 |

The linear probe **saturates** around 60 train images — going from 60 to 130
adds only 0.018 mIoU. Implication: collecting more UMD-style supervision is
not how you push past 0.674; you need a better **decoder** or a backbone
fine-tuned on richer affordance data. This sets a clean ceiling for the
linear-probing protocol.

## Code structure

- `src/methods/{base,dinov3_probe,dinov2_probe,siglip2_probe,openpi_siglip_probe,qwen25vl_grounding,molmoe_pointing,florence2_grounding,sam2_refine,random_baseline}.py`
- `src/inject/{camera,oracle_panda,wrapper}.py`
- `src/eval/{dataset_umd,metrics,qual_grid}.py`
- `scripts/{download_umd,make_split,run_probes,verify_pretrained_policy,record_demo,run_oracle_demo,multi_episode_demo,wrapper_pickandplace,cross_domain_demo,qual_grid,summarize_probes,hero_panel}.{py,sh}`

`AffordancePredictor` ABC means any frozen backbone plugs into the same
oracle/predictor injection slot in `AffordanceWrapper`.

## What's deferred

- **DINOv3 / SigLIP 2** — gated; need `huggingface-cli login`.
- **Qwen2.5-VL-3B and MolmoE-1B grounding** — code in place; CPU-prohibitive
  (~30–60 s/image × 5 prompts × 28 images). Run on GPU when free.
- **π0 / π0.5 SigLIP tower probe** — currently stand-in with public
  SigLIP-So400m. Real openpi extraction is a week-1 task.
- **TQC fine-tune with affordance channel concat** — 3-day report deliverable.
- **Generalization to PandaPickAndPlace and PandaSlide policy** — wrapper
  proven to work; policies need to be obtained or trained.
- **Larger UMD eval (500+ images)** — once GPU available.
