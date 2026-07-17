# AGENTS.md — working notes for AI-assisted sessions (paper on noise-robust RL-supervised GNC)

Context for Claude/agent sessions working on Carlo Zambaldo's MSc thesis follow-up paper.
Repo: RL-supervised ASRE+APF+SMC relative guidance for cislunar docking (github.com/CarloZambaldo/relativeGuidance, thesis in `TESI.pdf`).
Paper corrections doc: `..\..\paper_noise_corrections.md` (in the `TESI` folder).

## Thesis-reference configuration (do NOT change without reason)

These reproduce TESI.pdf Tables 5.5/5.10 and were restored on 2026-07-10 (commit `76e7549` and `7956311`)
after post-thesis commits (Jan–Apr 2026) had silently changed them:

- `freqGNC = 5 Hz` (was changed to 2 Hz post-thesis)
- Phase-2 sliding gains `[1, 2.11e1, 1]` (vel) + `8e-3` (pos); control law `u = u_opt − Umax·tanh(σ)` (Eq. 4.28, NOT `|σ|tanh(σ)`)
- APF: K_C_inside V-bar-dependent, cone `acone=0.02`, `bcone=10` (Table 4.4; post-thesis had 0.08/5)
- Terminal safe-mode handover in `RLEnvironment.step`: at range < 100 m force `AgentAction=2` and delete the ASRE
  reference. Without it, nominal mode misses the docking corridor in most regions even noiseless
  (leaving-aposelene 99% → 2%). It had been commented out post-thesis.

## Navigation noise model (added 2026-07-10 for the paper; thesis had NO nav noise — future work, p.54)

- First-order Gauss-Markov (τ = 60 s), scaled σ_r = noise%·min(range, 10 km), σ_v = noise%·min(‖v‖, 5 m/s)
- Constant-gain nav filter with control-feed-forward kinematic prediction
- Noise-adaptive dead-band on σ in OBGuidance (zero dead-band ⇒ exact thesis law).
  Rationale: without it, tanh noise-chasing burned 10–100× ΔV in nominal mode.
- Known physical limit: at 3% noise the 10 cm lateral docking corridor fails (~15–50 cm miss) because the
  lateral loop time constant ≈ 125 s freezes error ≈ 3%·v_dock·τ_lat ≈ 19 cm. Needs ≤1.5% noise or a better
  close-range sensor model to pass.

## Infra: MAGI servers (PoliMi)

- Hosts: casper / achiral / balthasar / melchior `.aero.polimi.it`, user `czambaldo`, ProxyJump `rainbow.aero.polimi.it`.
  Password-only auth (no SSH keys by choice); password in `C:\Users\carlo\.magi_pw`. Paramiko helper `magi.py`
  lives in the Claude scratchpad (recreate if missing: jump via rainbow, direct-tcpip channel to target).
- Homes NFS-shared (`/users/czambaldo` = `/home/czambaldo`); `/scratch` is per-machine.
- Sims run in rootless podman container `paiton:v01` (storage `/scratch/$USER/containers`; loaded on casper 2026-07-10).
  Repo mounted at `/code`, scratch at `/data`. `MonteCarlo_eval.py` falls back to `./Simulations` when `/data`
  is absent (local Windows runs).
- Agent models: `Agent_P1-v11.3.p1-multi-phase1-SEMIDEF` (named `Agent_P1-v11-thesis` on the servers, same zip),
  `Agent_P2-v11.5-multi-SEMIDEF`. Trained Python 3.12 / sb3 2.4.1; local Python 3.13 needs the
  `custom_objects` override in `MonteCarlo_eval.py` (else PPO.load segfaults).
- Standard MC invocation:
  `python MonteCarlo_eval.py -p <1|2> -m <model|_NO_AGENT_> -e <noise> -s 1753110 -n 100 -r False -x <region> -y`
  (`_NO_AGENT_` = safe mode; regions: aposelene, leaving_aposelene, approaching_aposelene, periselene)

## Paper MC campaign #1 — COMPLETE ✅

- Launched 2026-07-10 ~15:10 on **casper**, `run_campaign.sh` in tmux session CAMPAIGN; finished 2026-07-11 06:53
  (~15.5 h). 80 runs = 2 phases × 5 noise levels (0, 0.005, 0.01, 0.02, 0.03) × {agent, `_NO_AGENT_`} × 4 regions,
  100 sims each. All 80 saved a `.mat` in `~/main/relativeGuidance/Simulations/` on casper; per-run logs in
  `~/main/relativeGuidance/tmux_logs/` (an extra 81st log `MC_P1_N0_aposelene__NO_AGENT_.log` is a stale
  single-sim test, not part of the campaign). Verified 2026-07-16.

### Success counts /100 (from log outcome markers), per region apo / appr / leav / peri

Phase 1 (rendezvous; failures are all OUT_OF_TIME):

| noise | agent            | safe (`_NO_AGENT_`) |
|-------|------------------|---------------------|
| 0     | 92 / 93 / 93 / 99 | 100 / 100 / 100 / 100 |
| 0.005 | 92 / 92 / 92 / 82 | 99 / 99 / 99 / 35   |
| 0.01  | 89 / 86 / 90 / 46 | 87 / 87 / 87 / 22   |
| 0.02  | 43 / 48 / 40 / 29 | 47 / 48 / 47 / 16   |
| 0.03  | 26 / 33 / 25 / 15 | 27 / 26 / 27 / 15   |

Phase 2 (docking; failures are almost all CRASHED = corridor miss, consistent with the 19 cm lateral-miss limit):

| noise | agent             | safe (`_NO_AGENT_`) |
|-------|-------------------|---------------------|
| 0     | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 100 |
| 0.005 | 100 / 100 / 97 / 1 | 100 / 100 / 100 / 100 |
| 0.01  | 98 / 99 / 93 / 0  | 100 / 100 / 100 / 100 |
| 0.02  | 56 / 61 / 47 / 1  | 62 / 60 / 58 / 62   |
| 0.03  | 38 / 32 / 32 / 0  | 31 / 29 / 35 / 30   |

Notes:
- P2 periselene with agent ≈ 0% even noiseless is the thesis-known limitation (computeTOF/ASRE reference
  unsuited at periselene; alternative TOF formula is commented in the code) — not a regression.
- "ASRE did not converge: Singular matrix" lines in periselene logs are in-sim warnings, not crashes.
- The `OUT_OF_TIME` "RELATIVE DISTANCE" print norms all 6 state components (mixes pos+vel) — misleading, ignore.

## Session 2026-07-16: close-range sensor fix + Campaign #2 + paper writing

Goal: ≥99% P2 success in apo/leaving/approaching regions (user requirement; P2 was 30-60% at ≥2% noise).
Root cause analysis: terminal lateral miss ≈ p·v_dock·τ_lat (v_dock = 5 cm/s from `dockingState`, τ_lat = Kvel/Kpos = 125 s)
amplified 2-3× by the dead-band free-zone (db/Kpos ≈ 2σ_r). Fix (commit `9366218`):
- `nav_noise_sigmas()` in OBNavigation.py: shared σ model (injection + dead-band) with **docking-sensor handover**
  `f = clip(range/200 m, 0.05, 1)` on σ_r and σ_v. p=0 and long-range behavior unchanged; P1 mathematically unaffected.
- MonteCarlo_eval P2 tspan 0.033 → 0.045 (OOT no longer truncates the noisy TOF tail; P2 TOF ≈ 141±19 min vs old 206-min wall).
- Local validation n=10 each: P2 apo 3% agent 10/10, safe 10/10, leaving 1% agent 10/10 (were 38/31/93%); lateral miss 0.1-1.5 cm (was 33-38 cm).
- `analyze_MC.py` (repo root): aggregates the 1.7 GB .mat files ON casper (podman) into compact JSON
  (per-sim ΔV/TOF/texec/miss/final_err_m/min_err_m); run: `podman run --rm --entrypoint "" -v ~/main/relativeGuidance/:/code -w /code paiton:v01 python3 analyze_MC.py /code/Simulations /code/Simulations/MC_P?_summary.json`.
  NOTE: AgentActionHistory in .mat only records the forced handover actions (recompute counts unusable; use OBoTUsage).

State on casper (final, 2026-07-16 evening):
- **Campaign #2 COMPLETE** (10:52→17:39, 40 runs OK). Old P2 .mat archived in `Simulations/campaign1_P2_preCloseRange/`;
  campaign1 logs in `tmux_logs_c1/`. Swap .mat moved to `Simulations/agent_swap/` (collision-safe).
- **CAMPAIGN #2 RESULTS (P2)**: **100/100 success in apo/leaving/approaching at ALL noise levels 0-3%, BOTH modes** —
  user's ≥99% requirement exceeded. p95 lateral miss ≤1.3 cm at 3% (corridor 10 cm). Periselene: safe 100% everywhere,
  nominal 0-7% (thesis-known computeTOF limit; obot_frac≈1.0 = tracks flawed ASRE ref to crash).
  ΔV p=0 nominal 1.31 (apo) / 2.66 (leav) / 5.11 (appr) vs safe 9.23; TOF nominal 151-162 vs safe 137 min.
  **ΔV crossover ≈1% noise**: above it safe glide cheaper than nominal tracking (3%: 13.1 vs 18.5-27.7) — key paper finding.
- Swapped-agent (A2): P1-wrong ≈ identical stats to correct (3.85 vs 3.86 m/s, 92%); P2-wrong produces NUMERICALLY IDENTICAL
  trajectories to correct agent (same decision sequence: 1 compute + skip; obot 84%, rest = <100 m handover).
- Summaries `MC_P1_summary.json`, `MC_P2_summary.json`, `MC_swap_summary.json` downloaded to `paper_tesi_work/data/`.

Paper working copy: `TESI/paper_tesi_work/` (extracted from paper_tesi.zip; re-zip when done).
Done so far: noise-model section rewritten in 0_intro (GM + close-range handover eq + filter + pointer to dead-band);
1_methodology: κ gains values, 100-m terminal handover paragraph, new §Noise-Adaptive Dead-Band (sec:deadband, eq:deadband);
2_training: training paragraph incl. noiseless-training/out-of-distribution point; 3_simulations: IC descriptions fixed to code,
P1 success gate fixed 200 m→10 m, P1 tables generated (Tables/tab_P1_*.tex via `gen_paper_assets.py` + data JSONs), P1 discussion
written (safe≡ across apo regions is expected: same seed/ICs; failures at ≥2% are measurement-limited OOT loitering 35-60 m from gate);
nomenclature filled; `check_refs.py` label checker (only tab:dockOverrallPerfo pending until P2 assets).
ALL DONE (2026-07-16 evening): P2 tables+figures generated (Tables/tab_P2_*, Figures/MC_success_vs_noise.pdf,
MC_deltaV_vs_noise.pdf, MC_P2_terminal_dispersion.pdf), P2 section + conclusions written, A2 tables/prose regenerated,
abstract updated, nomenclature filled. Updated paper packaged as `TESI/paper_tesi_updated.zip` (original paper_tesi.zip untouched).
Regenerate assets anytime: `python gen_paper_assets.py data/MC_P1_summary.json data/MC_P2_summary.json` in paper_tesi_work/.
Note: stats cells show "--" when <10 successful runs (outlier-only statistics suppressed). No LaTeX locally/casper — compile on Overleaf.

### Key headline numbers (campaign1 P1, thesis-consistent)
- P1 apo p=0: safe ΔV 13.29±7.77 / TOF 210.7 min / 100%; nominal 3.86±2.07 / 113.2 min / 92% (−71% ΔV, −46% TOF).
- P1 periselene p=0.5%: safe 35% vs nominal 82% (agent replanning rescues periselene at low noise).
- texec (casper EPYC 7413 = 2.65 GHz base ×24 cores → paper GR740 conversion ×63.6): raw ~0.04-0.07 ms/step.

## Session 2026-07-16 (evening, continued): 2%-noise retraining experiment + explanatory plots

IMPORTANT CORRECTION from Carlo: the <100m terminal safe-mode handover is NOT an original thesis mechanism
(unlike what commit `76e7549`'s message says) — it was added post-thesis, in an earlier AI-assisted session.
Kept enabled by default for eval (still needed for the 100% P2 success results above), but an experiment is
underway to have the agent learn this behaviour itself instead of relying on the hardcoded override.

**Experiment**: retrain the P2 agent WITH 2% navigation noise (`Training.py -e 0.02`), and with the <100m
handover DISABLED during training (`terminal_handover_enabled=False`, auto-set whenever `-e` is given) so the
agent actually experiences and must learn the terminal approach itself, rather than having its action
overridden in the state that matters most. Code changes (commit `f3391be` + follow-ups):
- `env_config.py` / `RLEnvironment.py`: new `terminal_handover_enabled` param (default True — unchanged for
  MonteCarlo_eval/all eval scripts, which still force the handover).
- `Training.py`: `-e/--noise` sets `navigation_noise_percent` for training envs and auto-disables the
  handover; `-y` skips the interactive confirm prompt (needed for unattended tmux runs).

**Running**: tmux `TRAIN_P2_NOISE2` on **melchior** (not casper — casper was busy with campaign2/analysis;
melchior was idle). Podman image `paiton:v01` was not on melchior; transferred via `podman save` on casper →
NFS-shared home tarball (`~/paiton_v01.tar`, deleted after use) → `podman load` on melchior. Command:
`podman run --rm --entrypoint "" -v /home/czambaldo/main/relativeGuidance/:/code -w /code paiton:v01 python3 Training.py -p 2 -m Agent_P2-v12-noise2pct -e 0.02 -y`,
log `tmux_logs/TRAIN_P2_NOISE2.log`. 1.5e6 timesteps, 15 parallel envs, 24 threads; a prior similar P2 run
(1e6 steps) took ~4h on comparable hardware (inferred from file mtimes, no logged duration), so expect ~6h
(started 2026-07-16 19:43). A persistent Monitor polls every 20 min for "FINISHED TRAINING" or exceptions.
New model lands in `AgentModels/Agent_P2-v12-noise2pct/model/`.
Next steps once done: MC-eval this new agent (both handover on/off at eval time are worth trying) at
0/0.5/1/2/3% noise on aposelene/leaving/approaching, compare ΔV and success vs `Agent_P2-v11.5-multi-SEMIDEF`
— the interesting question is whether it learns a cheaper terminal strategy than the hardcoded safe-mode
handover, recovering some of the ΔV lost above the ~1% noise crossover documented in the paper.

**Explanatory plots** (Carlo wanted cone + example trajectories like in the thesis, referencing
`matlabScripts/MonteCarloPlots.m` + `plotConstraintsVisualization.m` — those MATLAB scripts use the WRONG
post-thesis cone params acone=0.08/bcone=5, not the restored thesis values 0.02/10 used here):
- `extract_trajectories.py` (repo root): pulls a handful of full trajectories out of a huge (1.7GB) MC .mat
  into a small JSON (position/velocity/control/agent-action/OBoT-usage histories), with `--stride` decimation
  and `--indices` for picking specific/paired sim ids (needed for fair same-IC comparisons across configs,
  since MC seeding makes sim index i the same IC across different noise/mode/model runs of the same region).
  Run via podman on casper (same pattern as analyze_MC.py), download via base64 over the magi.py SSH helper.
- `paper_tesi_work/gen_trajectory_plots.py`: renders 5 PNGs into `Figures/explanatory/` — cone+3 paired
  trajectories, terminal zoom, periselene failure case, one time-history breakdown, phase-1 KOS/Safe spheres.
  Gotcha found & fixed: matplotlib's `ax.set_box_aspect([1,1,1])` alone does NOT give true equal units per
  axis (it only shapes the drawn box, each axis autoscales independently) — use the `set_axes_equal_3d()`
  helper (matches limits+centers explicitly) for a non-misleading "axis equal" 3D plot. Also: the thesis cone
  flares very wide far from the target (physically correct — corridor is permissive far out, tapers hard only
  near contact; at V=-4.2km the half-width is ~5.5km, wider than the approach itself!), so rendering the
  full-length cone surface swamps the trajectories visually — only render it near-target (last ~500-600m) and
  scale the axes from the trajectory data, not the cone surface. Data extracted from campaign-2 P2 .mat files
  (aposelene paired sim ids 0,1,2 for the safe/nominal/noisy comparison; periselene for the failure case; a P1
  aposelene file for the sphere plot). All extracted data downloaded to `paper_tesi_work/data/traj_extracts/`.

Paper zip repackaged again as `paper_tesi_updated.zip` including `data/`, `gen_paper_assets.py`,
`check_refs.py`, `gen_trajectory_plots.py`. These 5 explanatory PNGs are exploratory renders sent directly to
Carlo for viewing, NOT (yet) inserted as paper figures — ask before adding, they'd need a clean caption
explaining the bounded-cone-rendering caveat above.

## Session 2026-07-17: full re-simulation with the noise-trained agent + complete paper rewrite

Carlo's explicit instructions (paraphrased): keep Phase 1 as-is (old agent, existing campaign-1 sims), but
relax the success margin in post-processing (10→30m, no re-simulation) to show the "failures" are noise-scale
near-misses, not crashes; for Phase 2, **re-simulate everything** (all 4 regions × 5 noise, nominal mode only
— safe mode is agent-independent, reused unchanged from campaign 2) with the new `Agent_P2-v12-noise2pct`
agent, delete the superseded old-agent (`v11.5`) P2 .mat files, and rewrite the whole paper accordingly with
an extremely detailed algorithm walkthrough, action-usage stats (SKIP/COMPUTE/DELETE %), updated plots, and a
redone NN-comparison appendix.

**Two real bugs found and fixed while extracting the "how many times does it use safe mode" data** (both
committed, both affect only diagnostic fields, not any previously-reported ΔV/TOF/success number):
1. `RLEnvironment.py`: the line writing the *actual* chosen action into `AgentActionHistory` was commented
   out — only the forced <100m override ever wrote to it. Fixed by uncommenting + coercing to a Python scalar
   (`int(np.asarray(AgentAction).reshape(-1)[0])`), since the value arrives as a numpy array from a VecEnv,
   not a plain int (crashed on first attempt, don't skip the coercion).
2. `analyze_MC.py`: SKIP/COMPUTE/DELETE fractions were normalized by a *fixed* max-duration decision count,
   silently understating every percentage for any episode that terminates early (i.e. every successful one).
   Fixed with a proper per-episode `n_decisions` field (`decisions.size` from the actual terminal index).
   Symptom before the fix: percentages summed to ~53% instead of 100%.
Net result after both fixes: SKIP≈83%, COMPUTE≈0.2% (essentially one ASRE solve per flight), DELETE≈16%
(**almost entirely the forced <100m override re-firing at every decision point while inside it, not
independent agent choice** — confirmed because both the "correct" and a "wrong" agent swapped into Phase 2
show the same ~73 deletes/episode, since the terminal glide is common infrastructure below 100 m regardless
of which agent supplied the pre-handover reference).

**Campaign 3** (`run_campaign3.sh` on casper, tmux `CAMPAIGN3`, 2026-07-17 09:57→~13:40, ~3.7h): 20 runs =
4 regions × 5 noise, `Agent_P2-v12-noise2pct`, nominal only, handover ON (production default), seed 1753110,
n=100. All 20 succeeded. **Headline: 100% docking success in aposelene/leaving/approaching at every noise
level 0-3% (exceeds Carlo's ≥99% target across all three, not just aposelene)**; periselene stays 0-8%
(unrelated known computeTOF limitation, confirmed unrelated to noise/agent-version). Also reran both
swap-agent tests fresh (`SWAP2_P1`, `SWAP2_P2` on melchior) so their action-logging uses the fixed code;
results archived to `Simulations/agent_swap_v2/` (shared NFS home, visible from any MAGI host).

**Key new finding — region-dependent efficiency, not a uniform crossover**: at aposelene the noise-trained
agent stays cheaper than safe mode at every noise level (barely crosses at p=3%: 12.22 vs 13.11 m/s). At
leaving/approaching aposelene, injecting *any* noise collapses the OBoT-usage fraction (time spent tracking
the optimal reference) from ~83% to 50-66%, i.e. the terminal safe-mode dwell time balloons once the estimated
range is noisy in these regions — this, not tracking-noise chatter, is what drives an early (p=0.5%) and
widening cost crossover there (up to +61% at p=3% at approaching aposelene). Compared to the OLD
(noiseless-trained) agent on the same regions: new agent is 13-37% *more* expensive at low noise (p≤1%,
more cautious policy) but 12-34% *cheaper* at p=3% in every region — a consistent trade, not a fluke.

Appendix A2 fully rewritten: the pre-retraining "agents are numerically interchangeable" story no longer
holds now that the P2 agent differs — new-P2-in-P1 costs 3.5x more (13.52 vs 3.86 m/s) via much more frequent
ASRE recomputation (3.75/episode vs ~1), while P1-in-new-P2 is *cheaper* than the correct agent (1.31 vs 2.59)
since it reproduces the old noiseless-optimized behavior the retraining traded away. NN weight comparison
(`compare_agents_nn.py`, local, uses the `custom_objects` PPO.load workaround) redone for P1-vs-new-P2 and
old-P2-vs-new-P2: both show the same pattern — a moderate (11-25%) fraction of first-hidden-layer neurons
retain cosine similarity across independently-initialized/trained networks, everything deeper is
uncorrelated, consistent with shared low-level feature extraction + fully task-specific deeper decision logic.

Cleanup done: old-agent P2 `.mat` files (20, ~32GB) and the old pre-fix `agent_swap/` deleted from casper
(shared home also removes from melchior/achiral/balthasar view); various scratch/temp analysis dirs removed.
`Simulations/` down to 130GB. AgentModels for both P2 agents (old v11.5 + new v12) kept (needed for the NN
appendix and for reproducibility).

Paper is now fully self-consistent top-to-bottom on the new agent (methodology, training, both results
sections, appendix, abstract, conclusions all rewritten/updated); `check_refs.py` clean. Regenerate everything
with: `python gen_paper_assets.py data/MC_P1_summary.json data/MC_P2_summary.json data/MC_P2_new_summary.json`
then `python gen_trajectory_plots.py`, both in `paper_tesi_work/`.
