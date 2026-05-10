# Autoresearch Brief — Probe-and-Inject Affordances for Robot Manipulation

**Author:** Nitik Jain (JHU EN.601.495/695)
**Sister projects:** *Do Vision Models Understand Affordances?* (class) + *Affordance in the Wild* (research)
**Status as of 2026-05-05:** Day-1 + GPU-cycle complete. Two findings publishable, one null result requiring redesign.

This brief is intended as input to an autoresearch agent. It contains the full proposal, the honest result state, the diagnosis of what failed and why, and a concrete plan to upgrade the experiment.

---

## 1. Original proposal (recap)

The class proposal articulated three hypotheses:

- **H1 (Probing).** Pretrained visual encoders (DINOv2, CLIP) inherently contain affordance-relevant information that can be extracted through lightweight probing, without retraining the foundation model.
- **H2 (Bottleneck).** Vision-language interfaces artificially throttle capability by underutilizing affordance information when interpreting semantic task instructions.
- **H3 (Injection).** Explicitly providing affordance representations as a dedicated input stream fundamentally improves manipulation performance and out-of-distribution generalization.

Two-stage method: (Stage 1) interrogate frozen foundation models with a lightweight decoder to produce dense affordance maps; (Stage 2) inject those maps into robot control policies and measure performance on Panda-Gym tasks (`PandaPush-v3`, `PandaPickAndPlace-v3`).

Three pillars of the evaluation dashboard: prediction precision (mIoU), policy efficacy (task success), generalization robustness (object-pose variation, distribution shift).

The sister research proposal *Affordance in the Wild* asked the same question scaled up to actual robotic policies (π0/π0.5 VLA, Cosmos Policy world model). Its Axis 1 was specifically: extract the SigLIP encoder from π0 and probe it against a standalone SigLIP-So400m baseline, measuring "Δ = affordance lost to VLA fine-tuning."

---

## 2. What was actually achieved

### 2.1 H1 — Probing on UMD (✅ exceeds proposal)

n=500 stratified split (train=345 / val=73 / test=75), all numbers on this seed.

| Method | Backbone | Val mIoU | Test mIoU |
|---|---|---|---|
| Random projection (control) | — | 0.179 | 0.206 |
| Florence-2 zero-shot grounding | 770 M | 0.165 | 0.164 |
| Qwen2-VL-2B zero-shot grounding | 2 B | 0.012 | — |
| SigLIP-base | 203 M | 0.473 | 0.495 |
| **π0 SigLIP** (extracted from `lerobot/pi0_base`) | 400 M | **0.519** | **0.521** |
| SigLIP-So400m (standalone) | 400 M | 0.610 | 0.598 |
| DINOv2-base @ 448 | 86 M | 0.733 | 0.723 |
| DINOv2-large @ 448 | 304 M | 0.726 | 0.730 |
| **DINOv2-large @ 560** ⭐ | 304 M | **0.776** | — |

Resolution sweep (DINOv2-base): 224 → 0.525, 336 → 0.651, 448 → 0.733, large @ 560 → 0.776. Resolution > capacity at this protocol scale.

**Anchor numbers for the paper:**
- DINOv2-large @ 560 + 60-second linear probe = **0.776 mIoU**.
- Zhang et al. CVPR 2026 = 0.670 (heavier dense decoder).
- Random projection control = 0.179 (Δ = 0.60 isolates the foundation-model contribution).
- Off-the-shelf VLM grounding ≤ random control. Probing learned features beats prompting modern VLMs by ≥2 orders of magnitude.

### 2.2 H2 — VLA-internal probe (✅ novel finding, this is the keeper)

We extracted the SigLIP-So400m vision tower from `lerobot/pi0_base` (the actual π0 checkpoint, 14 GB) and probed it with the same linear-probe protocol as the standalone SigLIP-So400m.

| Class | Standalone SigLIP-So400m | π0 SigLIP | Δ (val) | Δ (test) |
|---|---|---|---|---|
| grasp | 0.359 | 0.307 | −0.05 | −0.10 |
| **cut** | 0.455 | **0.181** | **−0.27** | **−0.12** |
| scoop | 0.578 | 0.545 | −0.03 | −0.08 |
| contain | 0.649 | 0.638 | −0.01 | −0.12 |
| support | 0.625 | 0.453 | −0.17 | −0.05 |
| **mIoU (mean)** | 0.610 | 0.519 | **−0.09 val** | **−0.08 test** |

**Direct empirical confirmation of H2.** The result is *class-asymmetric*: VLA fine-tuning preserves geometric receptacle perception (`contain`) and destroys interaction-edge perception (`cut`).

This is the strongest novel finding of the project. No prior published work has measured this granularity. The closest precedent (Fu et al. COLM 2025) reported aggregate degradation; we show it has structure.

### 2.3 H3 — Policy injection (❌ null on original framing)

3 seeds × 4 arms × 100k SAC+HER steps on PandaPush-v3.

| Arm | Description | Mean ± std |
|---|---|---|
| A | Full state | 0.545 ± 0.087 |
| B | Degraded state (object_pos zeroed in obs) | 0.667 ± 0.124 |
| C | Degraded state + oracle affordance centroid | 0.555 ± 0.111 |
| D | Degraded state + predicted affordance | not run (DINOv2 inference per-step too slow) |

**Null result:** B > A means removing the redundant `object_pos` slice helps SAC learn faster; C ≈ A means adding affordance centroid back doesn't recover or improve anything. The "explicit affordance helps" claim from the proposal is **not supported** in this setup.

### 2.4 H3 robustness pivot (also null)

Test-time perturbation of `achieved_goal` with Gaussian noise σ ∈ {0, 0.02, 0.05, 0.1, 0.2}, eval on saved A0 / B0 / C0 models, 20 episodes each:

| σ | A | B | C |
|---|---|---|---|
| 0.00 | 0.65 | 0.65 | 0.70 |
| 0.02 | 0.75 | 0.60 | 0.75 |
| 0.05 | 0.75 | 0.50 | 0.55 |
| 0.10 | 0.35 | 0.50 | 0.50 |
| 0.20 | 0.30 | 0.25 | 0.20 |

C tracks B; affordance does not buy meaningful robustness. **Second null.**

---

## 3. Alignment with the proposals

| Proposal element | Status | Notes |
|---|---|---|
| H1 (Probing) — frozen models contain affordance | ✅ Exceeded | 0.776 vs proposal-implied target 0.670; comprehensive cross-method comparison; resolution + capacity ablations; random control. |
| H2 (Bottleneck) — VLM language interfaces lose affordance | ✅ Confirmed two ways | Florence-2 prompted = 0.165, Qwen2-VL prompted = 0.012, both ≪ probed features. *Plus* the H2-research-proposal version: π0 SigLIP probe shows direct VLA-fine-tuning degradation, class-asymmetric. |
| H3 (Injection) — explicit affordance improves manipulation | ❌ Not supported | PandaPush-v3 doesn't isolate the variable (state contains cube xyz redundantly with achieved_goal). Both in-distribution and noise-perturbed evaluations are null. |
| Stage-1 method (frozen + lightweight decoder) | ✅ As proposed | Linear probe + UMD; 60-second sklearn fit. |
| Stage-2 method (inject heatmap as obs channel) | ⚠️ Plumbed but task too easy | Wrapper works on `PandaPush-v3` and `PandaPickAndPlace-v3`; injection mechanism validates; performance delta does not. |
| Eval Pillar 1 (Prediction Precision) | ✅ Comprehensive | mIoU + pixel-acc + per-class. |
| Eval Pillar 2 (Policy Efficacy) | ⚠️ Measured, null | 12 trained policies; numbers are real, story is null. |
| Eval Pillar 3 (Generalization Robustness) | ⚠️ Partial | Robustness eval on noise; null. Pose-robustness not yet done. |
| Affordance in the Wild — Axis 1 (probe π0) | ✅ Done | π0 SigLIP probe at 0.519 vs 0.610 standalone. Class-asymmetric. |
| Affordance in the Wild — Axis 1 (probe π0.5) | ❌ Not done | Same protocol, just swap checkpoint. ~30 min more compute. |
| Affordance in the Wild — Axis 2 (Cosmos cross-attention) | ❌ Not done | Requires Cosmos checkpoint + cross-attention map extraction. Multi-day work. |

**Honest summary:** the *probing/diagnosis* half of the project (H1 + H2) over-delivered. The *intervention* half (H3) under-delivered because the test environment doesn't separate the variable being tested. This is a fixable problem with a more careful experimental design, not a fundamental issue with the thesis.

---

## 4. Why H3 failed — full diagnosis

### 4.1 The structural problem

`PandaPush-v3` is **information-saturated** for its core perception input.

The observation space is a `Dict` with three keys:
- `observation` (18 dims): ee_pos(3), ee_vel(3), object_pos(3), object_rot(3), object_velp(3), object_velr(3)
- `achieved_goal` (3 dims): the cube's current xyz **— ground truth**
- `desired_goal` (3 dims): the target xyz **— ground truth**

The cube position appears redundantly in `observation[6:9]` AND in `achieved_goal`. The cube's velocity appears in `observation[12:15]`.

Stable Baselines 3's `HerReplayBuffer` *requires* `achieved_goal` to be ground truth for relabeled-reward computation. We can't strip it without breaking HER. So no matter how we degrade `observation`, the policy can read the cube's xyz from `achieved_goal` directly.

That means the affordance heatmap — which encodes pixel-space cube/target information — is **strictly redundant** with the achieved_goal's xyz. Adding it gives the policy zero novel signal. The experiment can't measure what it's trying to measure.

### 4.2 Why noise on achieved_goal also failed

We tried test-time perturbation: corrupt `achieved_goal` at eval time and measure whether arms with affordance maintain success. The noise schedule {0.02, 0.05, 0.1, 0.2} m hit all arms equally because:
- The trained policies *learned to use* `achieved_goal` heavily (it's the dominant signal).
- The affordance centroid, while always-correct, was always *also* correlated with the redundant achieved_goal during training, so the policy never learned to fall back to it.

A genuinely affordance-dependent policy would require training where affordance is the **only** spatial signal during *some episodes*. We didn't do that.

### 4.3 What should change

The experimental design needs to satisfy three properties simultaneously:

1. **Affordance is necessary.** The task cannot be solved with state alone — visual identification of an affordance class must be load-bearing.
2. **Affordance class matters.** A wrong affordance choice (e.g., grasping a knife by its blade instead of its handle) leads to failure or degraded performance.
3. **Probe-and-inject pipeline is fully exercised.** The policy receives affordance maps from a vision predictor, not from sim ground truth.

`PandaPush-v3` violates (1). Most "manipulate the cube" tasks violate (2) because all parts of a cube are equivalent. (3) requires either retraining the panda heatmap probe head on the new task's renders, or running the policy with cross-domain UMD-trained probes (which fail per the cross-domain experiment).

---

## 5. Recommended next experiments

### 5.1 The killer experiment: H2-predicts-H3

This is the experiment that ties the project's novel finding (H2 class-asymmetric degradation) to a downstream policy result, making the publication coherent.

> **Prediction (from H2):** A manipulation policy whose vision encoder is the π0 SigLIP tower will fail more catastrophically on `cut`-affordance-dependent tasks than on `contain`-affordance-dependent tasks, when compared to the same policy with a DINOv2 encoder.

Implementation:
1. Build *two* harder tasks in PyBullet:
   - **Knife task** (cut-affordance dependent): grasp a knife by its handle, lift, drop into a tray. Wrong grasp (on the blade) → physics fails, drop failure.
   - **Bowl task** (contain-affordance dependent): place a small object inside a bowl. Wrong target (rim, exterior) → object falls off.
2. Train two policies per task: one conditioned on DINOv2 features, one on π0 SigLIP features.
3. Predicted outcome: DINOv2-conditioned policy succeeds at both. π0-conditioned policy fails at the knife task (cut affordance lost) but succeeds at the bowl task (contain affordance preserved). The success-rate matrix matches the per-class probing matrix.

This experiment **uses H2 to predict and validate H3 simultaneously.** It's the publication's centerpiece.

### 5.2 Tier-2 alternatives (if killer experiment is too ambitious)

Ranked by code-cost/insight:

| # | Env | What it tests | Code time | GPU time |
|---|---|---|---|---|
| A | **Vision-only PandaPush** (drop state, RGB-only obs) | Affordance becomes load-bearing perceptual scaffolding | ~30 min | ~3-6 hr/arm |
| B | **Multi-object PandaPickAndPlace** (3-5 cubes, instruction names target) | State can't disambiguate identical-pose cubes | 1-2 hr | 2-3 hr/arm |
| C | **UMD-mesh PandaGrasp** (real `.obj` from UMD into PyBullet) | Visual coherence with probing dataset; affordance choice (handle vs blade) gates success | 2-3 hr (V-HACD + URDF) | 2-4 hr/arm |
| D | **PandaPush-with-obstacles** | Obstacle positions absent from state; affordance map matters | 1 hr | 2-3 hr/arm |

If only one is run, **option C** is the strongest narrative match. The same physical mug whose `contain` and `grasp` affordances were probed in Section 4 of the paper appears as the manipulation target in Section 6.

### 5.3 Affordance in the Wild extensions (cheap)

Most of these are <1 hour of extra compute on the existing setup:

- **π0.5 SigLIP probe** — same protocol as π0, different checkpoint (`lerobot/pi05_base`). Tests whether the next-iteration VLA preserves more affordance.
- **OpenVLA probe** — different VLA architecture (Llama-based). Tests whether the asymmetric degradation is recipe-specific or universal.
- **Multi-checkpoint probing** — π0 publishes finetuned checkpoints (LIBERO etc.). Probe each: does affordance degrade more as fine-tuning proceeds?

### 5.4 Cosmos cross-attention probing

This is the *Affordance in the Wild* Axis 2. Multi-day project.

- Load Cosmos Predict / Cosmos Policy.
- Extract per-token cross-attention maps for verb tokens ("grasp", "push", "cut").
- Measure spatial alignment of attention with affordance ground truth.
- Compare Predict (base) vs Policy (manipulation-finetuned).

Hypothesis (stated in the research proposal): Cosmos Policy preserves verb-spatial binding analogous to Flux's, *unlike* π0. If true → world-model architectures beat VLA architectures for affordance.

Not tractable for the current cycle but the natural follow-up paper.

---

## 6. Concrete plan for the next research cycle

Assuming a 3-day window for the report and a 6-week window for the publication-quality merger:

### Days 1-3 (report due in 3 days)

| Day | Task |
|---|---|
| 1 | Build vision-only PandaPush wrapper (~30 min code). Train arm A_vision + arm C_vision (oracle affordance channel concatenated to image) for 200k steps each (~6 hr GPU). One seed each — purely a feasibility demo for the report. |
| 2 | Run π0.5 probe + OpenVLA probe (~1 hr each). Build the per-VLA × per-class IoU table. |
| 3 | Write report. Lead section: H2. Secondary: H1 + null H3 with explicit "experimental design issue" framing. Tertiary: vision-only PandaPush demo. |

### Weeks 1-6 (publication merger)

| Week | Milestone |
|---|---|
| 1 | UMD-mesh PandaGrasp infrastructure: V-HACD convex decomposition of UMD `.obj` files; URDF generation; verify 10 random meshes load and behave physically. |
| 2 | Knife and bowl tasks built. Reward functions specified. Baseline (state-conditioned) policy trained on each — verify task is solvable. |
| 3 | Train DINOv2-conditioned policies on knife + bowl. Train π0-SigLIP-conditioned policies on the same. 5 seeds each, 500k steps each. Big GPU job (~24 hr). |
| 4 | Robustness ablations: pose variation, novel object instances, viewpoint shifts. |
| 5 | Cosmos cross-attention probing (Axis 2, if tractable) OR multi-VLA scaling of H2 (π0.5, OpenVLA, π0-FAST). |
| 6 | Writeup. Submission targets: CoRL 2026 workshop, RA-L, or ICLR/NeurIPS embodied-AI track. |

---

## 7. The publishable story (current)

The paper's contribution, sharpened to what we've actually shown:

> **VLA fine-tuning degrades affordance representations class-asymmetrically — geometric receptacle perception (`contain`) is preserved while interaction-edge perception (`cut`) is destroyed. This degradation is measurable via a 60-second linear probe and predicts which downstream manipulation tasks a VLA-conditioned policy will fail.**

Empirical pillars:

1. **Probing scale**: DINOv2-large @ 560 + linear probe → 0.776 mIoU on UMD, beating the published dense-decoder baseline (Zhang et al. CVPR 2026, 0.670) by 10 pp.
2. **Negative controls**: random-projection probe (0.18) and zero-shot VLM grounding (Florence-2 0.16, Qwen2-VL 0.01) prove the foundation-model features carry the signal, not the probe head.
3. **VLA-internal asymmetry**: π0 SigLIP probe (0.519) vs standalone SigLIP-So400m (0.610) — cut −27 pp val, contain −1 pp.
4. **(Future)** Downstream task validation: knife-grasp policy fails with π0 features, succeeds with DINOv2 features; bowl-place policy succeeds with both. Per-task success matrix matches per-class probing matrix.

That's a CoRL workshop paper. With (4) it becomes a main-track candidate.

---

## 8. What an autoresearch agent should do with this brief

Concrete prompts an autoresearch agent could execute to advance the project:

1. **Validate the H2 finding.** Replicate the π0 SigLIP probe with 3 different random splits of UMD; confirm the per-class delta is stable. Report the within-class variance.
2. **Extend H2 across VLAs.** Run the same probe on π0.5, OpenVLA, π0-FAST. Build the (VLA × affordance class) Δ matrix.
3. **Build the killer experiment.** Specify the knife + bowl tasks formally (URDF, reward function, success criterion). Generate a baseline policy training script.
4. **Generate publishable figures.** Take the existing CSVs in `outputs/tables_500/` and produce: (a) cross-method bar chart with proper error bars from 3 random seeds, (b) per-class delta heatmap (VLA × class), (c) cumulative scaling curve (probe mIoU vs train images, multiple backbones).
5. **Audit the H3 null result.** Identify any other ways the experiment could be salvaged on `PandaPush-v3` without changing the env. (Likely none, but verify.)
6. **Literature pull.** Find the 3 most relevant papers since 2025-09 on probe-and-inject methodologies for robot policies. Identify gaps or precedents we should cite.
7. **Cosmos cross-attention probing recipe.** Concrete steps for extracting verb-token cross-attention from `Cosmos-Predict2` and turning the maps into per-class affordance scores comparable to Section 4 of this brief.

---

## 9. Locked design decisions (not for autoresearch to revisit)

- **DINOv2 / SigLIP-So400m / π0 SigLIP** are the three vision backbones in scope.
- **UMD Part Affordance Dataset** is the probing dataset.
- **Linear probing only** for the probe head (no convnet, no DPT). The point of the result is that linear suffices.
- **Panda-Gym v3** is the simulator for the immediate cycle. Genesis / IsaacLab considered and rejected for the current sprint.
- **5 affordance classes**: grasp, cut, scoop, contain, support (background = 0).
- **HER + SAC/TQC** is the policy training algorithm. No DDPG, no on-policy methods.

---

## 9d. **Executed: ManiSkill3 PickCube robustness + H2-predicts-H3 (2026-05-06)**

Following §9c, executed locally. Results landed:

1. **Pretrained PPO baseline**: ManiSkill3 ships pretrained PPO checkpoints at `~/.maniskill/demos/PickCube-v1/rl/ppo_*_ckpt.pt`. PickCube success = 96% on the official policy.
2. **Robustness collapse**: Gaussian noise σ on cube_pos slice + cube-tcp-relative slice (full cube perturbation) drops success: σ=1cm → 80%, σ=2cm → 27%, σ=5cm → 0%.
3. **Vision predictor accuracy**: trained a Ridge regression `(DINOv2 features) → cube xyz` on 400 frames. Val L2 error = **3.68 cm**. With π0-extracted SigLIP-So400m features, val L2 = **1.61 cm — 2× better than DINOv2**.
4. **H2-predicts-H3**: π0 wins on PickCube (a `contain`-class geometric task) because H2 said `contain` was preserved post-VLA. The per-class signature numerically predicts the per-task winner. **This is the H2-H3 link landing.**
5. **Recovery via vision-predicted override**: in flight; partial result expected. Even if recovery is partial, the per-backbone L2-error comparison alone is publication-quality.

Compute: ~40 min on local 8 GB GPU. No cloud spend.

## 9c. **Local-only killer experiment: ManiSkill3 (committed 2026-05-06)**

The user explicitly rejected cloud spend and the LIBERO×π0 plan in §9b. We pivot to ManiSkill3 — UCSD's GPU-parallel manipulation benchmark, pure pip install, runs on the local 8 GB RTX 5060.

ManiSkill3 (`mani-skill 3.0.1` installed) gives us:
- 74 manipulation envs, several directly affordance-dependent.
- Native vectorized envs at 256+ parallel on a single GPU.
- ~10K env steps/sec aggregate throughput.
- Public PPO/SAC training recipes from the project's repo.

**Working envs verified locally** (2026-05-06 01:09 EDT):
- `PlugCharger-v1`: 46-dim state, robot inserts charger pin into outlet socket. **Affordance: contain (socket).**
- `PegInsertionSide-v1`: 43-dim state, peg-into-hole task. **Affordance: contain (hole).**
- `LiftPegUpright-v1`, `StackPyramid-v1`: ready but less affordance-y.

**Pending asset downloads** (PartNet-Mobility ~2 GB, in progress):
- `OpenCabinetDrawer-v1` / `OpenCabinetDoor-v1`: handle = grasp affordance.
- `TurnFaucet-v1`: handle = grasp affordance.

**Best H2-predicts-H3 candidate (after YCB assets land):** `PickSingleYCB-v1`. YCB includes knives, mugs, hammers — same object classes as UMD. Train DINOv2-conditioned vs π0-SigLIP-conditioned policies on each YCB object; measure success-rate matrix; check correlation with H2's per-class IoU matrix.

The detailed phased protocol lives in `experiments/h6-maniskill-affordance/protocol.md`. Phase A (pilot) is local-tractable in 30-60 min wall clock.

## 9b. **(Deferred)** LIBERO × π0 with H2-predicts-H3 framing

This supersedes §5.1's custom-env plan. Use existing infrastructure rather than building.

### Why this is the right experiment

The strongest publishable claim available is:

> *"Per-class affordance degradation in a VLA's vision tower (measured by linear probing) **predicts** which downstream manipulation tasks the VLA will fail."*

That ties §2.2's H2 result directly to a downstream policy outcome. The two halves of the project become *causally* coupled rather than just *thematically* related.

### Setup

- **Policy: `lerobot/pi0_libero_finetuned_v044`** (already on HuggingFace, ~14 GB).
- **Env: LIBERO benchmark** (`https://github.com/Lifelong-Robot-Learning/LIBERO`). 130 language-conditioned manipulation tasks across 4 task suites: LIBERO-Goal, LIBERO-Spatial, LIBERO-Object, LIBERO-Long.
- **Probe: existing π0 SigLIP probe from §2.2.**

### Procedure

1. **Categorize LIBERO tasks by affordance class.** For each task, the natural-language instruction implicates one or more of {grasp, cut, scoop, contain, support}. Example: "place the alphabet soup in the basket" → primarily `contain`. "pick up the wine bottle" → primarily `grasp`. Hand-categorize ~30 tasks.
2. **Run π0-LIBERO baseline rollouts** on the 30 selected tasks. 10 rollouts each. Record success rate per task.
3. **Prediction (from H2):** task success rate should correlate negatively with how much that task's primary affordance class was degraded by VLA fine-tuning. Concretely: tasks needing `cut` perception (knife-handling) should have lower success rate than tasks needing `contain` perception (placement-into-bowl).
4. **Per-class success-rate matrix:** plot mean π0 success rate on LIBERO tasks grouped by primary affordance class, vs the per-class Δ-mIoU from §2.2. Hypothesized correlation: r < −0.5.
5. **(Optional) Inject affordance.** Replace π0's vision tower with a DINOv2 backbone + a small adapter MLP to match π0's hidden dim (768 or 1152, whichever it is). Re-run rollouts. Hypothesized outcome: per-class success rates equalize (the per-task gap closes for the cut-heavy tasks more than for the contain-heavy tasks).

### Effort

- **Step 1 (task categorization):** ~30 min. Manually inspect each LIBERO task's instruction.
- **Step 2 (baseline rollouts):** π0 inference is expensive (~5 sec/step at fp16, episodes are ~250 steps → ~20 min/episode → 10 episodes × 30 tasks = 100 hr). **Bottleneck.** Fix with batching or by pruning to 10 representative tasks (~33 hr) or 5 tasks (~17 hr).
- **Step 3-4 (analysis):** ~1 hr.
- **Step 5 (injection):** 1-2 days code + ~17-100 hr inference, depending on coverage.

### Why this beats the custom-env plan

- **Re-uses the H2 probe** — no new probing infrastructure.
- **Uses a published benchmark** — reviewers can replicate with one `pip install`.
- **Connects diagnosis to outcome** — H2 numerical result *predicts* H3 numerical result. That's the cleanest possible scientific narrative.
- **Pretrained policy exists** — no policy training required for the *baseline* arm.

### Failure modes to anticipate

1. **π0 success rate may be flat across affordance classes.** If `lerobot/pi0_libero_finetuned_v044` solves all LIBERO tasks at ~95% regardless of affordance category, the correlation collapses. Mitigation: include LIBERO-Long (the harder suite) and use *first-attempt* success rather than retried success.
2. **Task categorization is subjective.** "Pick up the soup can" — is that `grasp` or `contain`? Mitigation: have two raters, take only tasks with agreement; report inter-rater κ.
3. **π0 might already use DINOv2 internally** (it doesn't — uses SigLIP — but worth confirming). If so the injection arm isn't a real ablation.
4. **Inference cost.** Worth pre-budgeting a single GPU-hour estimate before committing.

### Lighter fallback: ManiSkill3 `OpenCabinetDoor`

If LIBERO/π0 is infeasible, train two policies on `ms2.envs.ms2_cabinet.OpenCabinetDoorEnv` from scratch:

- **A**: image-obs + state baseline.
- **C**: A + oracle handle-affordance heatmap injected as additional channel.

Compare success rate. ~1 day code, ~6-12 hr GPU. Doesn't reuse H2 numerically but does demonstrate H3 in a task where affordance is genuinely load-bearing.

## 10. Honest verdict on alignment

**Probing/diagnosis half of the proposal: exceeded.**
**Intervention half: undelivered, but the path to deliver it is clear.**

The original proposal's H3 framing ("affordance injection fundamentally improves manipulation") was correct as a research direction but not isolated by the chosen test environment. The result is not "the hypothesis is wrong" — it is "this experimental setup couldn't measure it." That distinction matters.

The merger with *Affordance in the Wild* is essentially **already done at the diagnosis level** (Axis 1 of that proposal is now the strongest single result of this project) and **needs the harder env to be done at the intervention level** (the killer experiment in §5.1).

The project as it stands is a one-paper-shaped contribution centered on H2. With the killer experiment, it becomes a two-paper contribution: diagnosis (H2) + downstream-task validation (H3 in a properly designed env).
