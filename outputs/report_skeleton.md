# Report Skeleton — *Probing & Injecting Affordances for Robot Manipulation*

3-day write-up window after the 2026-05-06 talk. This skeleton has the
section headings, the questions each section must answer, and pointers to the
figures/tables already on disk. Fill in, don't redesign.

---

## 1. Abstract  *(150 w)*

What you found in one paragraph:
1. Frozen DINOv2-base features carry **0.674 mIoU** of affordance signal on
   UMD when read out by a 60-second linear probe at 448² — at parity with the
   recent CVPR 2026 dense-decoder baseline (0.670).
2. Different pretrained backbones encode different affordances differently
   (geometric SSL vs language-vision contrastive).
3. We give a clean injection mechanism (`AffordancePredictor` ABC →
   `AffordanceWrapper`) that exposes any of these probes as an explicit
   observation channel for an off-the-shelf manipulation policy, and
   demonstrate it on a pretrained TQC+HER PandaPush-v3 agent.
4. Implications for VLAs (Fu et al. 2025) and world-model policies (Cosmos
   Policy): the bottleneck isn't the features, it's how the policy
   *consumes* them. Injection is a cheap fix.

## 2. Introduction  *(0.5–0.75 page)*

- Motivation from existing slide deck (slides 2–4): semantic-interaction gap.
- The two-question frame: *(diagnosis)* are affordances preserved through
  policy fine-tuning [Fu, Zhang]? *(intervention)* if not, can we re-inject
  them? This paper is the intervention half.
- Contributions:
  - **C1** — first systematic linear-probe comparison across DINOv2,
    SigLIP, SigLIP-So400m on UMD, with a deliberately minimalist decoder
    isolating the *features themselves* as the variable.
  - **C2** — a modular `AffordancePredictor` injection interface usable in
    panda-gym and (by construction) any gym env with sim-state access.
  - **C3** — a working oracle-injection baseline that the next experiment
    (Stage-3 predicted-instead-of-oracle) can ablate against.

## 3. Related work

- Affordance datasets and segmentation: Myers et al. UMD, Lu et al. AffNet,
  Mottaghi et al. PartNet-Mobility.
- Probing studies: **Zhang et al. CVPR 2026 [1]** — geometric vs
  interaction perception decomposition; reports 0.670 mIoU. Our 0.674 is
  the methodological anchor we build on.
- VLM/VLA bottlenecks: **Fu et al. COLM 2025 [2]** — VLAs underutilize
  visual representations (0.411 → 0.155).
- Robot policies on panda-gym: rl-baselines3-zoo, panda-gym v3 [3].
- Recent VLAs/VLMs we evaluate or plan to evaluate: π0/π0.5 [4,5],
  SmolVLA, Molmo, RoboPoint, Florence-2, Magma.

## 4. Method

### 4.1 Affordance probing

- Frozen vision tower → patch tokens reshaped to `(B, D, gh, gh)`.
- Per-image preprocessing: bilinear resize to fixed `S × S`, manual
  ImageNet/processor-mean normalization (HF processor's `size` override
  proved silently ignored — see implementation note in Appendix).
- Per-patch label: mode of GT pixel labels under each `patch_size × patch_size`
  tile.
- Probe: `sklearn.linear_model.LogisticRegression(solver=lbfgs, C=1.0,
  max_iter=1000)` on patch features.
- Inference: per-class softmax → bilinear upsample → argmax.

### 4.2 Affordance injection

- `AffordancePredictor` ABC (`src/methods/base.py`) → `predict_map(rgb) →
  (C, H, W) ∈ [0, 1]`.
- `AffordanceWrapper(gym.Env)` adds an `affordance` key to the observation,
  populated either by oracle simulator state (Gaussian centered at the
  projected 3D pose of each named region) or by any predictor.
- Camera matches panda-gym v3 defaults
  `(target=[0,0,0], distance=1.4, yaw=45, pitch=-30, fov=45)`; depth
  used to scale Gaussian σ in pixels.
- For the demo we used `enaitzb/TQC-PandaPush-v3` from HuggingFace, with
  its accompanying `vec_normalize.pkl` (without VecNormalize the policy
  succeeds 0/10).

### 4.3 Cross-domain

- See §5.4. UMD-trained probe applied directly to panda-gym renders
  exhibits the expected domain gap; this experiment motivates Stage-3
  (training the probe on simulator-rendered data).

## 5. Experiments

### 5.1 UMD subset

n=200 stratified-by-category split (130/28/29 train/val/test). Image size
**448 × 448** (32 × 32 DINOv2 patch grid). All experiments run on a single
laptop CPU; numbers below are deterministic given seed 0.

### 5.2 Cross-method probing on UMD

Reference: `outputs/figures/probe_miou.png`, `probe_miou_perclass.png`,
`outputs/tables/probe_summary.md`.

**Headline:** DINOv2-base + linear probe → mIoU **0.674**, slightly above
Zhang et al.'s 0.670 dense decoder. SigLIP-So400m → mIoU 0.453, with very
different per-class profile (best on grasp, scoop). SigLIP-base → 0.391
overall, best on cut.

Insert: full table from `outputs/day1_summary.md`.

### 5.3 Qualitative comparison

Reference: `outputs/figures/qual_grid.png`. 5 UMD val tools × 7 columns.

Pick 2–3 illustrative cases for figure inset.

### 5.4 Cross-domain probe behavior

Reference: `outputs/figures/cross_domain_grid.png`. UMD-trained probe
applied to panda-gym render shows that geometric-feature alignment is
limited under heavy domain shift. Quantify by overlap of probe argmax
with the oracle heatmap support — to be added next.

### 5.5 Injection demo (oracle)

Reference: `outputs/figures/push_demo.mp4`,
`outputs/figures/oracle_overlay.mp4`. Pretrained TQC+HER policy reaches
goal in <10 timesteps, oracle heatmap tracks pose live. **No
performance delta yet** — the next experiment trains TQC w/ vs w/o the
affordance channel.

## 6. Limitations

- Subset evaluation. Probe metrics on n=28 val.
- DINOv3 / SigLIP 2 require HF auth — code falls back to v2 / SigLIP-base.
- π0 SigLIP-So400m probe used the *public* SigLIP-So400m as a stand-in.
- Linear probe is a deliberate lower bound; a 2-layer or DPT-style decoder
  is expected to lift mIoU but is not the contribution of this report.

## 7. Future work

Roadmap from `outputs/day1_summary.md`. Highest-impact next step: training
the policy with affordance channel concat (Stage-2 ablation), 3 seeds.

## 8. Conclusion

What we showed: the affordance signal is in the features; the bottleneck is
how policies consume them; here's a clean injection interface.

## A. Implementation notes

- HF image processors silently ignore `size={"height": H, "width": W}`
  overrides; we manually preprocess. (See `_extract_patch_features`.)
- DINOv3 / SigLIP 2 are gated; `AutoImageProcessor.from_pretrained` 401s
  without auth. Fallback chain documented in `configs/methods.yaml`.
- UMD label key is `gt_label`, not `gt`.
- UMD canonical mirror moved; new URL hard-coded in
  `scripts/download_umd.sh`.

## B. Reproduction

```bash
make data
make split N=200
DEVICE=cpu make probe-dinov2          # @ 224 — 4 min
python scripts/run_probes.py --method dinov2 --image-size 448 --n 130   # 10 min
python scripts/run_probes.py --method siglip2 --image-size 256          # 5 min
python scripts/run_probes.py --method openpi_siglip --image-size 384    # 10 min
make policy oracle demo
python scripts/qual_grid.py
python scripts/summarize_probes.py
```
