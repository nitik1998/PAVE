# H9 — Knife handle-vs-blade part discrimination

**Question**: when we restrict the affordance task to the binary
part-discrimination that the affordance-manipulation literature
(Aff-Grasp, TARAD, GIFT, Kokic '17) actually depends on, does the
VLA-induced "cut-class loss" we observed in H2 still show up?

**Setup**:
- UMD images restricted to {knife, shears, scissors, saw}: train=77, val=17, test=22.
- Pool patch labels (image_size=224, patch=14, gh=16). Keep only patches
  whose ground-truth label is grasp (handle, label=1) or cut (blade, label=2).
  Discard background, scoop, contain, support patches.
- Train per encoder a `LogisticRegression(class_weight='balanced')` on
  standardized patch features. Evaluate on the val and test images
  (objects disjoint from train).
- 8 encoders: DINOv2-base/large, standalone SigLIP-So400m, π0 SigLIP,
  π0.5 SigLIP, OpenVLA SigLIP, π0+adapter, π0.5+adapter.

## Results

### Test split (22 knife/shears/scissors/saw images)

| Encoder | val bal-acc | test bal-acc | test AUC | test F1 (blade) | feat dim |
|---|---|---|---|---|---|
| DINOv2-base | 1.000 | 1.000 | 1.000 | 1.000 | 768 |
| DINOv2-large | 0.971 | 1.000 | 1.000 | 1.000 | 1024 |
| standalone SigLIP-So400m | 1.000 | 1.000 | 1.000 | 1.000 | 1152 |
| **π0 SigLIP** | **1.000** | **0.985** | **1.000** | **0.985** | 1152 |
| π0.5 SigLIP | 1.000 | 1.000 | 1.000 | 1.000 | 1152 |
| OpenVLA SigLIP | 1.000 | 1.000 | 1.000 | 1.000 | 1152 |
| π0 + adapter | 0.971 | 0.971 | 1.000 | 0.970 | 256 |
| π0.5 + adapter | 1.000 | 1.000 | 1.000 | 1.000 | 256 |

**Every encoder solves the part-discrimination task at ≥97 % balanced
accuracy and AUC = 1.000.** π0 SigLIP — the encoder we said "lost cut
affordance" via the H2 per-class probe (cut IoU 0.181 vs standalone
0.455) — achieves test balanced accuracy 0.985 and AUC 1.000. The
adapter does not help (it slightly hurts on test, 0.971 — likely
overfitting on the small training set with 256-d compression).

## What this overturns about our earlier story

The H2 per-class IoU finding said π0 lost cut affordance by 27 pp on UMD.
We took that to mean π0 had lost the ability to perceive cut-class
regions. H9 shows that interpretation is **wrong**:

- π0 *can* tell handle from blade on a knife at 98.5 % accuracy.
- π0 *cannot* tell cut from {grasp, scoop, contain, support, bg} across
  17+ object types simultaneously with a single linear head as well as
  standalone SigLIP could.

These are different tasks. The first is what affordance-based
manipulation actually relies on (Aff-Grasp does grasp segmentation;
GIFT does keypoint identification; Kokic '17 does sub-volume affordance
scoring on a target object). The second is a much harder multi-class
confusion problem that mixes per-class semantics with per-object class
co-occurrence.

## What this means for the paper

Our project started from the H2 per-class IoU drop (π0 cut −27 pp,
contain −1 pp) and built a paper around "VLAs lose cut affordance and
the loss is recoverable." H9 shows that on the binary part-
discrimination metric the literature uses, π0 has not lost cut
perception — it just performs worse on UMD's specific multi-class
confusion benchmark.

The defensible reframing:

1. **Established by literature** (Aff-Grasp 77 % vs 46 % LOCATE; TARAD
   +52 %; GRAFF 3× faster RL convergence; FSD 72 % vs 42 %; GIFT 80 %
   on novel tools; AFFORD2ACT 82 %): per-part affordance perception is
   causally important for manipulation success.

2. **Our methodological contribution**: per-pixel multi-class probing
   metrics (UMD-style 5-class IoU, the benchmark the recent dense-
   decoder probe papers like Zhang et al. CVPR 2026 optimize) are a
   *narrow* lens. They penalize VLA encoders for class-confusion on
   diverse objects but do not track the binary part-discrimination
   task that actual manipulation pipelines depend on. On the
   manipulation-relevant metric, all the VLAs we tested (π0, π0.5,
   OpenVLA) match standalone SigLIP and DINOv2-large at ≥97 % balanced
   accuracy and AUC = 1.000.

3. **Open question** for future work: in cluttered scenes with multiple
   tool types simultaneously, does the multi-class probe's predicted
   degradation actually appear? Or does the per-tool binary
   discrimination keep being enough? This is an actual experiment we
   should propose as a follow-up.

## Files

- `experiments/h9-handle-blade/results/h9_handle_blade_results.json`
- `experiments/h9-handle-blade/results/h9_handle_blade.png`  — bar chart
- `outputs/figures/h9_qualitative_panel.png` — 5 test knives × 8 encoders overlay
- `experiments/h9-handle-blade/results/h9_test_pred_grids.npz` — raw prob grids
