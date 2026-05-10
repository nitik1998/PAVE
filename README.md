# PAVE — Probing Affordance in Vision-Language-Action Encoders

Class-asymmetric degradation, mechanism, and recovery of affordance
representations in deployed VLAs.

> Companion repository for the JHU EN.601.495 / 695 (Introduction to Robot
> Learning, Spring 2026) project by **Nitik Jain**. The project asks whether
> the per-class affordance signal carried by a foundation vision encoder
> (DINOv2, SigLIP-So400m) survives the end-to-end fine-tuning that produces
> a vision-language-action model (π₀, π₀.₅, OpenVLA), characterizes the
> mechanism of any loss, and tests a small intervention that recovers it.

## Headline results

All numbers from a 60-second `sklearn.LogisticRegression` linear probe on
top of frozen patch features, evaluated on UMD Part Affordance.

| Result | Number | Comparison |
|---|---|---|
| DINOv2-large @ 560² linear probe, mIoU on UMD val | **0.776** | Zhang *et al.* CVPR 2026 dense decoder = 0.670 (+10.6 pp) |
| π₀ SigLIP cut-class IoU drop vs. standalone SigLIP-So400m | **−27 pp** | contain-class drop = −1 pp; class-asymmetric loss |
| Cross-family check on OpenVLA SigLIP | **same pattern** | cut −22 pp, contain −2 pp; recipe-independent |
| Per-class drift → per-class IoU drop, Spearman ρ | **0.90** (p = 0.037) | n = 5 affordance classes; mechanism predicts behavior |
| 297K-parameter MLP adapter on frozen π₀, cut-class IoU recovery | **0.181 → 0.405** | 82% of the gap to standalone closed |
| Cut-class loss on binary part-discrimination (vs. multi-class IoU) | **−27 pp → −1.5 pp** | the multi-class probe overstates the deficit |

## Repository layout

```
src/                         Frozen-encoder probes + UMD dataset loader.
  methods/                   DINOv2, SigLIP, π₀, π₀.₅, OpenVLA, Florence-2,
                             Qwen2-VL, MolmoE, random-projection control.
  eval/                      mIoU / per-class IoU metrics, UMD splits.
scripts/                     Top-level run scripts.
  run_probes.py              Entry point: linear-probe a backbone on UMD.
  mechanism/                 Layer-wise CKA, drift, drift-IoU correlation.
  intervention/              MLP adapter training and evaluation.
configs/                     Affordance taxonomy YAML (5 + bg classes).
experiments/
  h6-maniskill-affordance/   ManiSkill3 PickCube probe → policy chain.
  h8-action-proxy/           Action-prediction proxy from frozen features.
  h9-handle-blade/           Single-object knife handle/blade discrimination.
  h10-multitool/             Cut-vs-rest across full UMD; multi-tool composites.
outputs/                     Tables, JSON results, headline figures.
slides/                      Beamer LaTeX deck for the 12-minute presentation.
paper/                       Paper scaffold + bibliography.
findings.md                  Living document of all measured results.
RESEARCH_ROADMAP.md          Tier-by-tier research plan (CoRL/RSS-targeted).
```

The 6 GB UMD dataset is **not** committed. See `data/README.md` for download
instructions.

## Reproducing the headline results

### 1. Environment

Conda is recommended.

```bash
conda create -n pave python=3.10 -y
conda activate pave
pip install -r requirements.txt
# PyTorch must match your CUDA toolkit; see https://pytorch.org/.
```

Tested on Python 3.10, PyTorch 2.4 + CUDA 12.x, scikit-learn 1.5,
transformers 4.45, timm 1.0.11. All experiments fit in 8 GB VRAM.

### 2. Data

```bash
bash scripts/download_umd.sh         # ~6 GB, ~10 min on a fast link
python scripts/make_split.py         # writes data/umd/splits_500/{train,val,test}.json
```

### 3. Linear probes

```bash
# DINOv2-large at 560², the SOTA-beating result
python scripts/run_probes.py \
    --method dinov2_large --device cuda --image-size 560 \
    --splits data/umd/splits_500 --pred-root outputs/predictions_500 \
    --tables-root outputs/tables_500

# π₀ SigLIP at native 224 — measures the class-asymmetric loss
python scripts/run_probes.py \
    --method pi0_siglip --device cuda --image-size 224 \
    --splits data/umd/splits_500 --pred-root outputs/predictions_500 \
    --tables-root outputs/tables_500

# Same for: pi05_siglip, openvla_siglip, openpi_siglip (standalone),
# dinov2, siglip2, florence2, random_features.
```

### 4. Mechanism analysis

```bash
# Layer-wise CKA + per-class final-layer drift (3 VLA families)
PYTHONPATH=. python scripts/mechanism/cka_with_openvla.py
# Drift→IoU correlation across all VLAs
PYTHONPATH=. python scripts/mechanism/drift_iou_all_vlas.py
# Bootstrap CIs on per-class IoU (n_boot = 1000)
PYTHONPATH=. python scripts/mechanism/bootstrap_iou_ci.py
```

### 5. Adapter recovery

```bash
# π₀ adapter
PYTHONPATH=. python scripts/intervention/adapter_recovery.py
# π₀.₅ adapter
PYTHONPATH=. python scripts/intervention/adapter_recovery.py --use-pi05
# OpenVLA adapter
PYTHONPATH=. python scripts/intervention/openvla_adapter.py
```

### 6. Complexity-spectrum analysis (H9 + H10a + H10b)

```bash
PYTHONPATH=. python experiments/h9-handle-blade/knife_part_discrimination.py
PYTHONPATH=. python experiments/h10-multitool/h10a_cut_vs_rest.py
PYTHONPATH=. python experiments/h10-multitool/h10b_composite_scenes.py
```

### 7. Slides

```bash
cd slides && pdflatex deck.tex && pdflatex deck.tex
```

## Key findings, in one paragraph

VLA fine-tuning rotates per-class affordance directions out of a single
linearly-readable hyperplane in a class-asymmetric way. The rotation is
mechanistically localized to the final transformer block (layer-wise CKA
drops from 0.95 to 0.02 across the last layers in π₀), and the per-class
magnitude of the rotation numerically predicts the per-class IoU drop
under linear probing (Spearman ρ = 0.90, p = 0.037). A 297K-parameter MLP
adapter on top of frozen features recovers most of the lost signal,
demonstrating that the information was rotated rather than deleted.
However, on a complexity spectrum that ranges from multi-class probing to
binary part-discrimination — the formulation downstream manipulation
pipelines actually use — the apparent 27 pp deficit collapses to under
3 pp. The standard multi-class probing protocol overstates the encoder
deficit relative to the perception sub-problem manipulation actually
solves.

## Limitations

- All probing claims are on UMD Part Affordance. Cross-dataset validation
  (AGD20K, IIT-AFF) is in scope but was not completed.
- No closed-loop manipulation evaluation. The natural follow-up — LIBERO
  × π₀ per-task success-rate prediction from per-class probe scores —
  requires ≥ 24 GB VRAM for π₀ inference.
- The MLP adapter recovers the per-pixel UMD probe but does not transfer
  cleanly to multi-tool composite scenes.
- Mechanism correlation has only n = 5 classes per VLA. Significant for
  two of three VLAs.

See `findings.md` for the complete list of caveats.

## Citing

This work is unpublished as of May 2026. If you build on it, please cite:

```bibtex
@misc{jain2026pave,
  author       = {Nitik Jain},
  title        = {PAVE: Probing Affordance in Vision-Language-Action Encoders},
  year         = {2026},
  howpublished = {Project for JHU EN.601.495/695, Spring 2026},
  note         = {\url{https://github.com/nitik1998/PAVE}}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

UMD Part Affordance dataset (Myers *et al.*, ICRA 2015). LeRobot project
(Hugging Face) for the π₀ / π₀.₅ open-weight checkpoints. OpenVLA
(Kim *et al.*, CoRL 2024) authors for the open-weight Prismatic-Llama
checkpoint. ManiSkill3 (Tao *et al.*, ICLR 2025) for the simulation
environment and pretrained PPO baseline.
