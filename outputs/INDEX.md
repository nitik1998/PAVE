# Index of Outputs

Top-level navigation for the project deliverables. **Updated 2026-05-06 night** with mechanism, intervention, and cross-family results.

## Latest (CoRL-grade, post-CKA + adapter + OpenVLA)

- [`../findings.md`](../findings.md) — **single source of truth for paper claims** (mechanism + recovery + OpenVLA cross-family)
- [`../RESEARCH_ROADMAP.md`](../RESEARCH_ROADMAP.md) — research-roadmap planning doc, target = CoRL 2026
- [`../paper/main.tex`](../paper/main.tex) — **8-page CoRL paper scaffold**, builds against `outputs/figures/*.png`
- `mechanism/cka_and_drift.png` — layer-wise CKA + per-class final-layer drift
- `mechanism/drift_vs_iou.png` — drift→IoU correlation, Spearman ρ=0.90 for π0
- `mechanism/drift_vs_iou.csv`, `mechanism/correlation_summary.json`
- `mechanism/per_class_drift.json` — per-class cosine sim (final layer)
- `bootstrap/h2_per_class_ci.png` + `bootstrap/bootstrap_summary.csv` — per-class IoU 95% CIs (n_boot=1000)
- `figures/h2_h5_h7_delta.png` — **4-VLA per-class IoU bars**: standalone / π0 / π0.5 / OpenVLA
- `figures/adapter_recovery.png` — **hero figure**: linear vs MLP adapter on 3 VLA encoders
- `intervention/adapter_pi0_siglip_h256.json`, `adapter_pi05_siglip_h256.json`, `adapter_openvla_siglip_h256.json`
- `tables_500/openvla_siglip_overall.csv`, `tables_500_test/openvla_siglip_overall.csv`

## Documentation
- [paper_results.md](paper_results.md) — **paper-quality consolidated results (H1, H2, H3 plan, negative results)**
- [talk_outline_v2.md](talk_outline_v2.md) — **15-min talk for n=500 + π0 + H3 results**
- [day1_summary.md](day1_summary.md) — Day-1 narrative summary
- [final_results.md](final_results.md) — earlier comprehensive doc (n=200)
- [talk_outline.md](talk_outline.md) — earlier talk outline (n=200)
- [report_skeleton.md](report_skeleton.md) — 3-day report writing scaffold
- [results.json](results.json) — machine-readable consolidation of all `_overall.csv` rows

## H6 / H5 (newest, paper-quality)

- `figures/h6_robustness_pickcube.png` — pretrained PPO collapses 96→27→0 with cube_pos noise.
- `figures/h6_predictor_quality.png` — π0 SigLIP cube_pos predictor 1.61cm vs DINOv2 3.68cm (consistent with H2 contain-preserved).
- `figures/h6_recovery_pickcube.png` — recovery sweep: oracle 100%, vision overrides 0% (state-redundancy makes naïve override fail).
- `figures/h6_pickcube_demo_pi0.mp4`, `h6_pickcube_demo_dinov2.mp4` — 60-frame 512×512 rollouts with predictor cube_pos overlay (true=green, pred=red).
- `figures/h6_hero_panel.png` — combined H6 hero (16:9).
- `figures/h2_h5_delta.png` — 3-VLA per-class IoU (standalone / π0 / π0.5). Shows asymmetric degradation + π0.5 partial recovery.
- `outputs/h6_results.md` — narrative of the H6 chain.

## Headline figures (paper-quality, n=500)
- `figures/probe_miou_n500.png` — **6-bar val+test mIoU comparison @ n=500**
- `figures/probe_perclass_n500.png` — **5-class IoU bars across all backbones @ n=500 test**
- `figures/h2_delta.png` — **H2 result: π0 SigLIP vs standalone, per-class delta (val + test panels)**
- `figures/hero_demo_4k.mp4` — **720p × 3-panel pretrained-TQC demo** (3 successful PandaPush episodes)

## Earlier headline figures (n=200)
- `figures/probe_miou.png` — val mIoU bar chart (8 methods)
- `figures/probe_miou_perclass.png` — per-class IoU bars (val)
- `figures/probe_miou_test.png` — test mIoU bar chart
- `figures/probe_miou_test_perclass.png` — per-class IoU bars (test)
- `figures/qual_grid.png` — 5 UMD val tools × 7 methods
- `figures/qual_grid_test.png` — 5 UMD test tools × 4 methods
- `figures/scaling_curve.png` — DINOv2 mIoU vs n_train (saturates @ ~60)
- `figures/cross_domain_grid.png` — UMD-trained probe applied to PandaPush renders
- `figures/hero_panel.png` — single 16:9 PNG combining mIoU + per-class + qual_grid

## Demo videos
- `figures/hero_demo_4k.mp4` — **paper-quality 720p × 3-panel composite (3 episodes)**
- `figures/push_demo.mp4` — pretrained TQC + oracle heatmap (1 episode)
- `figures/push_demo_multi.mp4` — 3 successful episodes back-to-back with title cards
- `figures/push_pretrained.mp4` — clean pretrained policy rollout (no overlay)
- `figures/oracle_overlay.mp4` — heatmap-only motion demo (12 frames random actions)
- `figures/pickandplace_overlay.mp4` — same wrapper, no changes, on PandaPickAndPlace-v3
- _(after H3 sweep)_ `figures/h3_arms_4panel.mp4` — 4-arm side-by-side trained policies

## Tables (raw)
- `tables_500/*_overall.csv`, `tables_500_test/*_overall.csv` — **paper-quality val+test @ n=500**
- `tables_500/probe_summary_n500.md` — n=500 markdown summary
- `tables/*_overall.csv`, `tables_test/*_overall.csv` — earlier n=200 results
- `tables/probe_summary.md`, `tables_test/probe_summary.md` — n=200 markdown tables
- `tables/scaling_curve.csv` — train-size scaling
- `tables/c_ablation.csv` — regularization sweep (run via `scripts/c_ablation.py`)
- _(after H3)_ `outputs/h3/sweep_results.csv`, `outputs/h3/summary.json` — policy training results

## Predictions
- `predictions/{method}/{id}.npy` — per-image (C, H, W) val predictions
- `predictions/test/{method}/{id}.npy` — per-image test predictions

## Quick re-runs
```bash
# All figures from current state:
python scripts/build_all_figures.py

# Or specific:
python scripts/summarize_n500.py
python scripts/plot_h2_delta.py
python scripts/plot_h3_curves.py
python scripts/dump_results_json.py
bash scripts/asset_index.sh
```

## H3 (policy injection) — running

Live status: see `outputs/h3_sweep.log`. Per-arm-seed snapshots in
`outputs/h3/{A,B,C,D}/seed*/{train_log.csv,eval.json}`.

Sweep config: 4 arms × 3 seeds × 100k steps SAC+HER on PandaPush-v3.
Predictor for arm D: `outputs/checkpoints/panda_heatmap_head.joblib`
(Ridge regressor over DINOv2-base patch features, R² ≈ 0.6 per channel,
fit on 200 random Panda renders).

After completion, run `python scripts/plot_h3_curves.py` to generate
`outputs/figures/policy_curves.png` and `policy_final_bar.png`.
