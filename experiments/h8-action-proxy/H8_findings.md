# H8 — Action-prediction proxy (does affordance help downstream?)

**Question**: do encoders that preserve affordance predict policy-relevant
quantities better than encoders that don't?

**Method**: a two-task crossover.
- Contain task: predict the pretrained ManiSkill3 PickCube PPO's actions
  from RGB features (n_train=1702, n_test=426).
- Cut task: predict knife / shears / scissors orientation (blade→handle
  unit vector) and handle / blade centroids on UMD images
  (n_train=77, n_val=17, n_test=22).
- Per encoder, mean-pool patch features → Ridge regression → mean L2 error.

## Headline numbers

### Contain task — PickCube action prediction (mean L2, lower better)

| Encoder | Feat dim | Mean L2 | Median L2 |
|---|---|---|---|
| **pi0 + random projection to 256** | **256** | **0.933** | **0.884** |
| pi0_adapter (UMD-trained, hidden=256) | 256 | 0.936 | 0.870 |
| pi05_adapter | 256 | 0.963 | 0.896 |
| standalone + random projection to 256 | 256 | 1.039 | 0.950 |
| dinov2_large | 1024 | 1.087 | 1.000 |
| pi0_siglip | 1152 | 1.094 | 1.034 |
| dinov2_base | 768 | 1.101 | 1.014 |
| pi05_siglip | 1152 | 1.213 | 1.090 |
| openvla_siglip | 1152 | 1.311 | 1.254 |
| standalone_siglip | 1152 | 1.378 | 1.272 |

### Cut task — UMD orientation and centroid prediction

| Encoder | Orient val | Orient test | Handle L2 | Blade L2 |
|---|---|---|---|---|
| **pi0_siglip** | **28.0°** | **33.9°** | **0.094** | **0.082** |
| dinov2_base | 30.0° | 30.1° | 0.128 | 0.090 |
| dinov2_large | 34.8° | 54.3° | 0.158 | 0.092 |
| pi05_siglip | 40.0° | 37.6° | 0.137 | 0.086 |
| pi0_adapter | 46.8° | 46.3° | 0.132 | 0.109 |
| standalone_siglip | 39.0° | 48.0° | 0.160 | 0.090 |
| openvla_siglip | 42.4° | 56.4° | 0.167 | 0.097 |
| pi05_adapter | 56.8° | 61.7° | 0.195 | 0.108 |

## What the experiment actually shows

### The original premise was wrong

We expected: **encoders with degraded cut-class IoU (π0, OpenVLA) would
predict cut-task quantities worse than encoders with preserved cut IoU
(DINOv2, standalone SigLIP)**. The opposite is true.

π0 SigLIP — the encoder we said "lost" the cut class via probing — is the
**single best encoder for cut-task orientation prediction** (28° val,
34° test, vs DINOv2-base 30°/30°, vs standalone SigLIP 39°/48°). The
adapter trained to recover the linearly-readable cut signal **hurts** on
the cut-task: pi0_adapter is 46° vs raw pi0_siglip 28°.

### The dim-reduction confound is fatal

On the contain task, the affordance-trained adapter (pi0_adapter, 0.936)
appears to beat raw pi0 features (1.094). But a **random projection of
the same pi0 features to 256-d achieves 0.933** — statistically
indistinguishable from the adapter. The "win" is not affordance-specific;
it is just dim reduction.

Standalone SigLIP-So400m at full 1152-d is the worst encoder on PickCube
(1.378), but with a 256-d random projection it jumps to 1.039 — a 25 %
drop in error from dim-reduction alone. The Ridge regression with
n_train=1702 simply overfits 1152 features.

### What this rules in / rules out

**Rules out**:
- The strong claim "VLA fine-tuning destroys affordance and the destruction
  manifests as worse downstream prediction" is **not supported** by these
  experiments.
- The claim "adapters trained for per-pixel affordance recovery improve
  downstream action prediction" is **not supported** beyond a generic
  dim-reduction effect.

**Rules in**:
- VLA fine-tuning shifts encoder geometry in a way that hurts
  *per-pixel linearly-readable* cut signal but preserves or even improves
  *image-level* cut-task content. This is consistent with the per-pixel
  probe being a *narrow* lens on the encoder's signal.
- Encoder choice does affect downstream prediction error by 30–50 %
  (cut task: 28° to 62° spread; contain task: 0.93 to 1.38 spread). So
  encoder choice matters, but not in the direction the per-pixel probe
  predicts.

## Implications for the paper

1. **The story has to change**. The original narrative was "VLAs lose
   affordance → adapter recovers affordance → adapter helps downstream."
   The first arrow is true at the per-pixel probe level only. The second
   arrow is true at the per-pixel probe level only. The third arrow has
   no evidence — and where there appeared to be evidence (PickCube), it
   is fully explained by random dim-reduction.

2. **The interesting finding is now methodological**: per-pixel linear
   probing systematically under-reports VLA-encoder usefulness for
   downstream image-level tasks. This is an important caveat for the
   recent wave of probing papers (Voltron, R3M, SiamCLR-Robot, etc.) that
   benchmark robotic encoders by per-pixel or per-patch probing.

3. **The original "affordance is required for manipulation" premise is
   neither supported nor refuted by these experiments**. We tested action
   prediction on a contain task (PickCube) and orientation/centroid
   prediction on a cut-affordance object. Neither test forced the policy
   to discriminate between two affordance regions on the same object —
   the sense in which "affordance is required" would actually bite.

## What would actually answer the original question

A task where:
- Two regions on the same object require different actions (knife handle
  vs blade — grasp the handle, do not grasp the blade).
- The reward distinguishes these (positive for handle grasp, zero or
  negative for blade grasp).
- The policy receives only image-level features and must localize the
  correct region.

PickSingleYCB-v1 with knife / hammer assets in ManiSkill3 is the closest
locally-runnable analogue. Training PPO from scratch on it on 8 GB has
been unreliable in our setup. The cleanest local path is to **build a
synthetic part-aware reaching task** in ManiSkill3 with a knife, where
reward is +1 for end-effector → handle centroid and 0 otherwise. We have
not built this.

## Bottom line for the user

We do **not** have evidence that affordance is causally required for
downstream manipulation in the tasks we tested. We have evidence that
encoder choice matters by 30–50 % on downstream prediction error, but the
ordering of encoders does not match the ordering predicted by per-pixel
affordance probing. The original probing-and-injection thesis needs
either:
- a different downstream task (part-aware reaching);
- a different probing metric (image-level, not per-pixel); or
- a reframing where the *probing* result is interesting on its own
  (encoder geometry diagnostics), without claiming downstream
  manipulation consequences.

Files: `experiments/h8-action-proxy/results/{pickcube,umd_cut}_action_results.json`,
`outputs/figures/h8_crossover.png`, `h8_pareto.png`.
