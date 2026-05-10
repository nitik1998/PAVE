# Tomorrow's 15-Minute Talk — Slide-by-Slide Outline

Each section below: **what to say**, **what to show** (asset file in `outputs/`), and **what NOT to claim**.

---

## (1) Title — 30 s
> *"Do Vision Models Understand Affordances? Probing and Injecting Affordance Representations for Robot Manipulation."*
- One-sentence thesis: **Modern frozen vision foundation models already encode affordances; we extract them with a 60-second linear probe and inject them as observation channels into a pretrained Panda-Gym policy. No backbone training. No policy training.**
- Show: existing slide 1 from the deck.

## (2) Problem framing — 90 s
- Slide 2 from deck: affordances bridge pixels and physical action.
- Slide 3 from deck: VLAs degrade affordance (Fu et al. 2025: 0.411 → 0.155); world models may preserve it. *Diagnosis* — that's the sister project.
- Slide 4 from deck: semantic vs interaction gap.
- Reframe: **two-proposal frame**. Diagnosis (other proposal) + intervention (this class project). Today's talk = the intervention half.

## (3) Method overview — 90 s
- Slide 6/7 from deck: H1/H2/H3, two-stage method.
- **Add a slide**: the intervention is just an `AffordancePredictor` ABC. Any frozen backbone can plug in. Oracle simulator state is one such predictor; pretrained DINOv2 is another.
- "No fine-tuning anywhere. Linear probing + drop-in observation channel."

## (4) Probing results — 3 min
- **Show**: `outputs/figures/probe_miou.png` — 5-bar val mIoU chart, +1 random-baseline.
- **Headline number**: **DINOv2-base + linear probe at 448² = mIoU 0.674 on UMD val** (n=28, fit on 130 train, 60-sec `sklearn.LogisticRegression`). Slightly **above** Zhang et al. CVPR 2026 (0.670) which used a heavier dense decoder.
- **Random-projection control**: 0.229 — proves the foundation-model features (not the probe head) carry the signal. Δ = 0.445.
- **Florence-2 zero-shot grounding**: 0.052 — off-the-shelf VLM grounding fails on affordance segmentation by an order of magnitude.
- **Test-split sanity**: DINOv2 0.674 → 0.613 on n=29 unseen images. SigLIP-So400m climbs to 0.604, nearly tied. Per-class profiles diverge:
  - DINOv2 wins **grasp**, **scoop**, **contain**.
  - SigLIP-So400m wins **cut**, **support**.
- Show `outputs/figures/probe_miou_test.png` if there's time.
- **Data-efficiency**: `outputs/figures/scaling_curve.png` — the probe **saturates at ~60 UMD train images** (0.656 → 0.674 from 60 to 130). The bottleneck is the decoder, not the data.
- **Show**: `outputs/figures/probe_miou_perclass.png` — per-class IoU.
- **Cross-backbone story** (this is the talk's intellectual hook):
  - DINOv2 (geometric SSL) wins `contain` (0.627) and `support` (0.692). Geometric receptacles, surfaces.
  - SigLIP-So400m (large language-vision contrastive) wins `grasp` (0.261) and `scoop` (0.449). Action-relevant parts.
  - Different backbones encode different *types* of affordance — direct empirical hook for the *Affordance in the Wild* probing thesis.
- **Don't claim**: that DINOv2 universally beats everything. Per class, the picture is mixed.

## (5) Qualitative cross-method — 2 min
- **Show**: `outputs/figures/qual_grid.png` — 5 UMD val tools × {RGB, GT, dinov2, dinov2_448, siglip2, openpi_siglip}.
- Pick one example to talk through (the bowl in row 1): all backbones detect the receptacle, but at 448 the mask becomes tight; SigLIP-So400m's mask leaks into the rim.
- Pick the trowel (row 3): DINOv2 highlights the scoop region in green; SigLIP-So400m highlights it more strongly. Confirms the per-class story.

## (6) Injection mechanism + Panda-Gym demo — 3 min
- **Show**: architecture diagram from existing slide 10.
- **Show video**: `outputs/figures/push_demo_multi.mp4` — 3 successful pretrained-TQC PandaPush episodes back-to-back with oracle heatmap side panel and title cards.
- **Show video** (if time): `outputs/figures/oracle_overlay.mp4` — heatmap follows object/goal pose live as the cube moves.
- **Show frame**: `outputs/figures/pickandplace_overlay_03.png` — same wrapper, no modification, on PandaPickAndPlace-v3 (secondary task in proposal). Demonstrates wrapper genericity.
- "The wrapper preserves the original Dict obs space, so the pretrained TQC sees its training-time inputs. The affordance channel is exposed for downstream affordance-aware policies — that's the next experiment."
- **Don't claim**: a success-rate delta with vs without affordance. Not measured tonight.

## (7) Cross-domain failure + limitations — 60 s
- **Show**: `outputs/figures/cross_domain_grid.png`. UMD-trained DINOv2 probe applied zero-shot to PandaPush renders paints the entire white tabletop as `support` and the navy wall as `contain`. **This is the failure mode that motivates Stage 3** (predicted-instead-of-oracle in sim, fine-tune on simulator-rendered pixels).
- n = 28 val / 29 test images (small but stratified). We will scale to 500+ on GPU.
- DINOv3 / SigLIP 2 gated on HF — fell back to v2 / SigLIP-base. Code falls back gracefully and CSVs honestly record `actual_backbone`.
- π0 SigLIP-So400m probe used the *public* SigLIP-So400m as a stand-in; the actual π0-fine-tuned tower is a week-1 experiment via openpi.
- All numbers are CPU runs. A4 minutes per probe with `OMP_NUM_THREADS=4`.
- UMD-trained probes have not been tested on Panda-Gym renders (cross-domain gap). That's its own experiment.

## (8) 6-week roadmap + publication target — 90 s
| Week | Deliverable |
|---|---|
| 1 (DONE TODAY) | Probing infra, oracle injection, pretrained TQC demo, 4-backbone comparison |
| 2 | HF auth → DINOv3 + SigLIP 2. GPU → Qwen2.5-VL & MolmoE & Florence-2 |
| 3 | Continue-train TQC w/ affordance channel (3 seeds) |
| 4 | Replace oracle with predicted heatmap. Cross-domain UMD→Panda gap study |
| 5 | Probe SigLIP encoder extracted from π0 / π0.5 (research-project tie-in) |
| 6 | Writeup. Submission target: CoRL 2026 workshop track or RA-L |

## (9) Closing — 60 s
> "We treat this as a research-engineering question, not a benchmark question. The contribution is the framework: a single `AffordancePredictor` interface that lets ANY frozen backbone plug into a manipulation policy as an explicit affordance prior. Today: 4 backbones, real numbers, working PandaPush rollout. Next 6 weeks: VLA-internal encoders, the actual injection-arm ablation, publication."

---

## Asset checklist (everything is in `outputs/`)

| Slide | Asset |
|---|---|
| 1 | existing deck |
| 2-4 | existing deck |
| 5 | architecture from existing deck |
| 6-7 | hypothesis/blueprint from existing deck |
| 8 (new) | `outputs/figures/probe_miou.png` |
| 9 (new) | `outputs/figures/probe_miou_perclass.png` |
| 10 (new) | `outputs/figures/qual_grid.png` |
| 11 | architecture from existing deck slide 10 |
| 12 (new) | `outputs/figures/push_demo.mp4` |
| 13 (new, optional) | `outputs/figures/oracle_overlay.mp4` |
| 14 | `outputs/day1_summary.md` (this doc references it) |
| 15 | roadmap text only |

## What to print and bring

- A single laptop-ready folder with `outputs/figures/*.png` and `*.mp4`.
- The CSVs in `outputs/tables/` for live Q&A on individual numbers.
- This `talk_outline.md` as your speaker notes.
