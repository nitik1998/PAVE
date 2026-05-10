# Research Roadmap — Affordance, VLA Encoders, and Manipulation

**Goal:** turn the current class-project results into a CoRL/RSS-tier publication that lays the groundwork for a deeper Affordance-in-the-Wild research line. Author: Najib (Northeastern → JHU).

**Date opened:** 2026-05-06.

## What we already have (status 2026-05-06)

| Hypothesis | Headline number | Strength |
|---|---|---|
| H1 — probing UMD | DINOv2-large @ 560 → mIoU **0.776** (beats Zhang et al. CVPR 2026 0.670 by +10.6 pp) | Strong, novel |
| H2 — VLA encoder degradation | π0 SigLIP probe = 0.519 (Δ −0.09 vs standalone), class-asymmetric: cut −0.27, contain −0.01 | Novel, mechanistic flavor |
| H5 — recipe dependence | π0.5 partially recovers cut (0.18 → 0.26), recipe-dependent | Novel, narrative-strong |
| H6 (predictor) — H2 predicts geometry | π0 SigLIP cube_pos predictor 1.61cm vs DINOv2 3.68cm on PickCube | Strong (causally interesting) |
| H6 (recovery) | Vision-override fails (oracle 100% / vision 0%) due to redundant state slices | Honest negative, generalizable |
| H3 — affordance injection in PandaPush | Null (HER's `achieved_goal` saturates info channel) | Honest null |

## What's missing for a top-tier publication

A CoRL/RSS paper needs at least 3 of:

1. **Breadth of evidence**: ≥ 3 VLAs, ≥ 2 affordance datasets, ≥ 2 downstream tasks.
2. **Mechanism**: an interpretable account of *why* the degradation is class-asymmetric.
3. **Causal connection**: representation property numerically *predicts* downstream behavior.
4. **Intervention**: a fix that recovers the lost capability — adapter, fine-tuning recipe, or architectural change.

Current state: we have **1**(partial), **3**(partial), and we have not done **2** or **4**.

## Plan — 4-tier execution

### Tier 1 — strengthen what we have (this session, all local 8 GB GPU)

**T1.A — Layer-wise CKA (mechanism)**
Compute Centered Kernel Alignment between standalone SigLIP-So400m and π0/π0.5 SigLIP, layer by layer, on UMD test set. Hypothesis: divergence concentrates in late layers and on edge/handle features (cut), not on receptacle features (contain). This produces the first mechanistic plot of the paper.

**T1.B — Per-class representation drift (mechanism)**
For each affordance class, compute mean cosine similarity between standalone-SigLIP and π0-SigLIP patch features at pixels labeled with that class. Predicts a per-class drift metric that should correlate with the per-class IoU drop (T1.A is layer-axis, T1.B is class-axis).

**T1.C — Bootstrap confidence intervals**
n_boot = 1000 on per-class IoU for each backbone. Standard practice for ML conferences. Lets us state significance of the asymmetric pattern.

**T1.D — OpenVLA SigLIP probe (breadth)**
OpenVLA uses a fused DINOv2 + SigLIP-base vision encoder (1024-dim per stream → 2048 fused). Extract via `vision_backbone.featurizer` keys; run probe on UMD. If class-asymmetric pattern holds in OpenVLA (different family, different fine-tuning corpus from π0), the result generalizes. Already cached; just need to download blobs.

### Tier 2 — connect to downstream + intervention (this session)

**T2.A — Knife-handle disambiguation (cut-task downstream)**
Tiny task: given a cropped UMD knife/shears RGB, predict which end is the handle (binary classification). Linear classifier over patch features. π0 SigLIP should lose to standalone here. Pairs with H6 predictor (contain task → π0 wins) for the **per-task crossover** that closes H2 → H3.

**T2.B — Adapter recovery (intervention)**
Train a 2-layer MLP adapter on top of frozen π0 SigLIP features against UMD's cut-class labels. Show that ~10K parameters recovers most of the 27 pp lost. Direct positive intervention result.

**T2.C — Robot adapter on knife task**
If T2.A + T2.B work, plug the adapter into the H6 predictor framework on a knife-affordance task. Establishes that fixing perception fixes downstream.

### Tier 3 — additional benchmarks (this session, optional)

**T3.A — AGD20K cross-dataset validation**
AGD20K (CVPR 2022): 20K images, 36 affordance classes. Run the same probe protocol; show asymmetric VLA degradation persists. Either downloads from BU mirror or HuggingFace dataset.

**T3.B — RGB-D Part Affordance (3D grounding)**
Adds depth dimension to validate the geometric receptacle-vs-edge distinction in 3D.

### Tier 4 — paper packaging (this session)

**T4.A — CoRL paper scaffold**
8-page LaTeX skeleton with all sections; fill abstract, intro, method, experiments. Reuse `outputs/findings.md` as the source of truth.

**T4.B — Final figures**
Hero figure with the H2 → H6 → recovery story. Confidence interval bars. Per-class IoU heatmaps across VLAs.

**T4.C — Reviewer-proof story**
Anticipate "you only ran one π0 — could be a checkpoint quirk" → answered by Tier 1.D and Tier 3.A. Anticipate "linear probes are toy" → answered by Tier 2.A linking probe to downstream task. Anticipate "PickCube is too easy" → answered by Tier 2.C and the H6 robustness collapse.

## Order of operations (this session)

1. **T1.A + T1.B** — mechanism plots (parallel; 30 min)
2. **T1.C** — bootstrap CIs (parallel; 5 min CPU)
3. **T2.A** — knife-handle task (40 min including data prep)
4. **T2.B** — adapter recovery (30 min training + 10 min eval)
5. **T1.D** — OpenVLA download + probe (background; 1 h)
6. **T3.A** — AGD20K download + probe (background; 1 h)
7. **T4.A** — paper scaffold + new figures (30 min)
8. **T4.C** — reviewer-anticipating paragraphs in findings.md

## Target venue analysis

- **CoRL 2026** (deadline ~Jul 1): perfect fit — manipulation, learned representations, encoder analysis. Submit there.
- **RSS 2026** (deadline ~Feb 1): missed; aim for 2027 if we extend.
- **NeurIPS 2026** (deadline ~May 14): possible but ML-focused; reviewers may want broader empirical scope.
- **ICRA 2027** (deadline ~Sep): backup if CoRL rejects.

## Hardware budget

- All Tier 1, Tier 2, Tier 4 work runs on RTX 5060 8 GB (current local box).
- Tier 3 AGD20K may need ~10 GB scratch for image cache; have 98 GB free.
- No cloud spend.
