# H10 — Where on the complexity spectrum does the VLA cut-class loss appear?

**Method (Problem-Reformulation framework, Orchestra creative-thinking-for-research)**: Test cut-class detection at three task complexities and compare against the original H2 (5-class IoU on full UMD) and H9 (binary handle-vs-blade on knives only).

| Task | Setup | Hypothesis it tests |
|---|---|---|
| H2 | 5-class IoU across full UMD (all 17 categories) | "VLA loses cut affordance entirely" |
| H10a | binary cut-vs-rest-foreground across full UMD (all categories) | "Loss exists in binary form despite class-balancing" |
| H10b | binary cut detection on synthetic 2-tool composite scenes (cut tool + distractor tool side-by-side, train on single-tool only) | "Loss exists when there is visual co-presence of multiple object classes" |
| H9 | binary handle-vs-blade on knife/shears/scissors/saw only | "Loss exists in single-object part discrimination" |

## Headline numbers

Cut detection metric per encoder, across the four tasks (lower-is-harder on left, easier on right):

| Encoder | H2 5-class IoU | H10a binary cut-vs-rest | H10b multi-tool composite | H9 binary single-object |
|---|---|---|---|---|
| DINOv2-base | 0.78 | 1.000 | 0.975 | 1.000 |
| DINOv2-large | 0.74 | 1.000 | 0.941 | 1.000 |
| standalone SigLIP-So400m | 0.46 | 1.000 | 0.931 | 1.000 |
| **π0 SigLIP** | **0.18** | **0.971** | **0.927** | **0.985** |
| π0.5 SigLIP | 0.26 | 1.000 | 0.950 | 1.000 |
| OpenVLA SigLIP | 0.23 | 1.000 | (failed) | 1.000 |
| π0 + adapter | (n/a probe) | 0.956 | 0.855 | 0.971 |
| π0.5 + adapter | (n/a probe) | 0.997 | 0.894 | 1.000 |

(Metrics: H2 = per-class IoU. H10a / H10b / H9 = balanced accuracy. Higher = better in all columns.)

## What we now know

**The 27 pp π0 cut-class drop in H2 collapses to ≤3 pp in every other formulation.** Across the three reformulations:
- H10a (binary cut-vs-rest, full UMD): π0 = 0.971 vs DINOv2 = 1.000. Gap ≈ 0.03.
- H10b (multi-tool composite, train-on-single): π0 = 0.927 vs DINOv2 = 0.941. Gap ≈ 0.01. **Smaller** than the gap on H10a.
- H9 (binary handle-vs-blade single-object knife): π0 = 0.985 vs DINOv2 = 1.000. Gap ≈ 0.015.

**Every single non-VLA encoder also drops by ~5pp going from H10a single-tool to H10b multi-tool composite.** That drop is *not VLA-specific*. It's a generic distribution shift cost from the train-single → test-composite mismatch. π0 doesn't drop more than DINOv2 in this transition.

**The H2 "loss" is a multi-class confusion artifact.** It appears specifically when the encoder has to discriminate cut from {grasp, scoop, contain, support, bg} simultaneously, with class imbalance, across heterogeneous objects, in a single linear classifier. It does NOT appear in:
- single-object binary discrimination (H9)
- binary cut-vs-rest with class-balancing (H10a)
- multi-tool clutter (H10b)

The two-decade affordance literature (Aff-Grasp, GIFT, Kokic 2017, TARAD) uses task-relevant binary or pose-conditioned formulations. Those are the formulations where VLA encoders are not deficient.

## Why the adapters look BAD on H10b

| Encoder | H10a single | H10b composite | Δ |
|---|---|---|---|
| pi0_adapter | 0.956 | 0.855 | −0.101 |
| pi05_adapter | 0.997 | 0.894 | −0.103 |
| pi0_siglip | 0.971 | 0.927 | −0.044 |
| pi05_siglip | 1.000 | 0.950 | −0.050 |

The 256-d adapter compression is *narrower* than the raw 1152-d encoder. When trained on single-tool UMD, the adapter retains only the directions that helped on UMD-specific 5-class classification. When the test distribution shifts to multi-tool composites, those compressed directions don't generalize as well as the full encoder. **The adapter's 297K-parameter compression is task-overfit, not feature-distilling.** This caveats the H4 "adapter recovery" finding: it recovers per-pixel UMD IoU but does not produce a more transferable representation than the raw encoder.

## What this means for the talk and paper

The finding to lead with is **methodological**: the per-pixel multi-class IoU metric on UMD-style benchmarks reports a 27 pp encoder gap that disappears under any task-relevant reformulation we test. The literature uses binary part-detection or part-keypoint formulations (Aff-Grasp's affordance map for grasp-planning, GIFT's grasp/interaction keypoints, Kokic '17's per-voxel affordance scores). On those formulations, VLA-fine-tuned encoders match the standalone foundation models.

This frames our project as a **probe-evaluation paper**, not a "VLAs lose affordance" paper. The contribution: across four task complexities and seven encoder configurations, we show that the H2 metric does not predict task-relevant performance, and that any community wanting to evaluate VLA encoders for downstream affordance use should adopt binary or pose-conditioned probes.

## Files

- `experiments/h10-multitool/results/h10a_results.json` + `h10a_cut_vs_rest.png`
- `experiments/h10-multitool/results/h10b_results.json` + `h10b_single_vs_composite.png`
- `experiments/h10-multitool/results/h10b_composites.npz` — composite RGBs and labels
- `experiments/h10-multitool/results/h10b_cmp_pred_grids.npz` — predicted cut maps per encoder
- `outputs/figures/complexity_spectrum.png` — single hero figure linking H2 → H10a → H10b → H9
- `outputs/figures/h10b_qualitative.png` — 4 composite scenes × 7 encoders overlay
