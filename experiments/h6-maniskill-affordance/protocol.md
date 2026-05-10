# Experiment H6 — ManiSkill3 affordance injection (local-only, no cloud)

**Status:** scaffolding ready, not yet executed.
**Date set up:** 2026-05-06 01:09 EDT.
**Owner:** Nitik Jain.
**Compute:** RTX 5060 8GB (local). No cloud cost.

## Why this experiment

H3 on PandaPush-v3 was null because the env is information-saturated (HER's `achieved_goal` carries ground-truth cube xyz). H3' on LIBERO×π0 was deferred because π0 inference needs ~24 GB VRAM (we have 8 GB) and renting cloud is off-limits.

ManiSkill3 (UCSD, GPU-parallel) gives us:
1. Tasks where affordance choice gates success (`PegInsertionSide-v1`, `PlugCharger-v1`, `OpenCabinetDrawer-v1`, `OpenCabinetDoor-v1`, `TurnFaucet-v1`, `PickSingleYCB-v1`).
2. Native parallel envs on a single GPU — 256 parallel envs at <2 GB VRAM, ~10K steps/sec total throughput.
3. Pure-python install, no cloud, no asset downloads for the simplest tasks.

## What "affordance" means in each task

| Env | Affordance class (UMD label) | Why it matters |
|---|---|---|
| `PegInsertionSide-v1` ✅ ready | `contain` (the hole) | Robot must align peg with hole position. |
| `PlugCharger-v1` ✅ ready | `contain` (outlet socket) | Charger pin into socket; precise alignment. |
| `OpenCabinetDrawer-v1` (PartNet downloading) | `grasp` (handle) | Wrong grasp position → no force on drawer. |
| `OpenCabinetDoor-v1` (PartNet downloading) | `grasp` (door handle) | Same as drawer. |
| `TurnFaucet-v1` (PartNet downloading) | `grasp` (faucet handle) | Same. |
| `PickSingleYCB-v1` (YCB downloaded) | `grasp` (object-specific) | YCB knife → handle vs blade; mug → handle vs body. **Direct H2-predicts-H3 link.** |

**Best H2-predicts-H3 candidate: `PickSingleYCB-v1`.** YCB has knives, mugs, hammers — exactly the object classes whose `cut`/`grasp`/`contain` affordances we measured in H1/H2. If the policy fails more on knife-grasp tasks (cut affordance) than mug-grasp tasks (contain) when conditioned on π0 features vs DINOv2 features, the H2 → H3 prediction lands.

## Protocol (committed)

### Phase A — pilot (verify infrastructure)
1. Install: `pip install mani-skill` ✅ done.
2. Verify two envs load on local GPU: ✅ done. `PegInsertionSide-v1` (state=43-dim) and `PlugCharger-v1` (state=46-dim) both work.
3. Write a minimal vectorized PPO training script for `PlugCharger-v1` with state observation only. Train 200K steps × 256 parallel envs = ~80M env steps total. Should hit >70% success rate based on ManiSkill3 baselines.
4. Estimated wall-clock on RTX 5060: 30–60 min.

### Phase B — affordance injection
1. Re-run PlugCharger-v1 with three observation arms:
   - **A**: state only (43-46 dim).
   - **C-oracle**: state + 2-channel oracle Gaussian centered on outlet/socket pose (from sim ground truth) projected to camera pixels, pooled to (u, v) centroid + peak intensity = 6 extra dims.
   - **D-predicted**: state + 2-channel predicted heatmap from a fresh probe head trained on ManiSkill3 renders.
2. 3 seeds each, 200K steps each. Compare mean success rate.

### Phase C — affordance class × backbone (the killer test)
Once `PickSingleYCB-v1` works (after YCB asset confirmation) and PartNet assets land:
1. Group YCB objects by primary affordance class:
   - **`grasp`-class:** screwdriver, knife (handle), spatula.
   - **`contain`-class:** mug (interior), bowl, pitcher.
   - **`scoop`-class:** spatula (concave side), spoon.
2. Train two policies on each group:
   - **DINOv2-conditioned:** image obs → DINOv2-base patch features → policy MLP.
   - **π0-SigLIP-conditioned:** image obs → π0 SigLIP-So400m patch features (extracted from `lerobot/pi0_base`) → policy MLP.
3. Hypothesis (from H2): π0-conditioned policy fails more on `grasp`/`cut`-dependent objects than `contain`-dependent objects. DINOv2-conditioned policy succeeds across the board.
4. The success-rate matrix (π0/DINOv2 × affordance class) should mirror the per-class IoU matrix from H1/H2.

This is the publication's centerpiece — if and only if the asymmetry shows up.

## Compute budget (all local)

| Phase | Wall-clock estimate (RTX 5060, 8 GB) |
|---|---|
| A pilot (PlugCharger state-only PPO) | 30–60 min |
| B injection comparison (3 seeds × 3 arms) | 5–9 hours |
| C killer test (DINOv2 vs π0 SigLIP × N affordance classes × seeds) | 20–40 hours |

ManiSkill3's GPU-parallel architecture means 256 concurrent envs × 8 GB VRAM is feasible. PPO update is the bottleneck, not env stepping.

## Code skeleton

`experiments/h6-maniskill-affordance/code/train_h6.py` will be:
1. Import `mani_skill.envs`, build vectorized `PlugCharger-v1`.
2. CleanRL-style PPO loop (single-file, ~300 lines).
3. Optional `--obs-mode {state, state+oracle, state+predicted}` flag for arm switching.
4. Optional `--vision-backbone {none, dinov2, pi0_siglip}` for Phase C.
5. Save per-step metrics to `experiments/h6-maniskill-affordance/results/{arm}/seed{n}.csv`.

## Risks

1. **PPO doesn't converge in 200K steps on PlugCharger.** Mitigation: ManiSkill3 baselines hit ~80% in 200K, this is well-trodden territory.
2. **PartNet/YCB asset downloads stall.** Mitigation: use only PlugCharger and PegInsertionSide which need no extra assets.
3. **DINOv2/π0 SigLIP per-step inference still too slow.** Mitigation: pre-extract features once per episode (cache), use a smaller backbone (DINOv2-small @ 224 instead of large @ 560).
4. **Affordance redundant with state again.** This is the failure mode of Phase A. If state is sufficient for PlugCharger, switch to image-obs for Phase B/C. ManiSkill3's `obs_mode='rgb'` returns 128×128 RGB images.

## Success criteria

For the publication, we need:
1. **Phase B**: arm C (oracle affordance) ≥ arm A by ≥10 pp success rate, p < 0.05 across seeds.
2. **Phase C**: π0-vs-DINOv2 success-rate gap correlates with H2's per-class IoU delta (Spearman ρ > 0.5).

If Phase B is null but Phase C is not, that's still publishable — the policy doesn't gain from added oracle info but it gains from higher-quality features. If both are null, that's a hard refutation of H3.
