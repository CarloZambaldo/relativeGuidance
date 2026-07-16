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

State on casper (2026-07-16 ~11:10):
- **Campaign #2 RUNNING**: tmux CAMPAIGN2, `run_campaign2.sh`, P2-only 10 configs × 4 regions = 40 runs, log campaign2.log, ~8 h est.
  Old P2 .mat archived in `Simulations/campaign1_P2_preCloseRange/`; campaign1 logs in `tmux_logs_c1/`.
- **SWAP_P1 / SWAP_P2 tmux**: swapped-agent reruns (P1 with Agent_P2, P2 with Agent_P1-v11-thesis, aposelene p=0) to regenerate
  paper Appendix A2 numbers. WARNING: their .mat files land in Simulations/ and would collide with campaign entries in analyze_MC's
  (region,noise,mode) keying — move them to `Simulations/agent_swap/` BEFORE re-running the aggregations.
- `MC_P1_summary.json` (campaign1 P1, valid) downloaded to paper repo `paper_tesi_work/data/`.

Paper working copy: `TESI/paper_tesi_work/` (extracted from paper_tesi.zip; re-zip when done).
Done so far: noise-model section rewritten in 0_intro (GM + close-range handover eq + filter + pointer to dead-band);
1_methodology: κ gains values, 100-m terminal handover paragraph, new §Noise-Adaptive Dead-Band (sec:deadband, eq:deadband);
2_training: training paragraph incl. noiseless-training/out-of-distribution point; 3_simulations: IC descriptions fixed to code,
P1 success gate fixed 200 m→10 m, P1 tables generated (Tables/tab_P1_*.tex via `gen_paper_assets.py` + data JSONs), P1 discussion
written (safe≡ across apo regions is expected: same seed/ICs; failures at ≥2% are measurement-limited OOT loitering 35-60 m from gate);
nomenclature filled; `check_refs.py` label checker (only tab:dockOverrallPerfo pending until P2 assets).
TODO when campaign 2 completes: move swap .mat → agent_swap/, rerun analyze for P1(unchanged)+P2, download MC_P2_summary.json,
run `gen_paper_assets.py` (tables+3 figures), write P2 prose + conclusions (5_conclusions.tex is EMPTY), update A2 tables from
swap results, re-zip paper_tesi.zip.

### Key headline numbers (campaign1 P1, thesis-consistent)
- P1 apo p=0: safe ΔV 13.29±7.77 / TOF 210.7 min / 100%; nominal 3.86±2.07 / 113.2 min / 92% (−71% ΔV, −46% TOF).
- P1 periselene p=0.5%: safe 35% vs nominal 82% (agent replanning rescues periselene at low noise).
- texec (casper EPYC 7413 = 2.65 GHz base ×24 cores → paper GR740 conversion ×63.6): raw ~0.04-0.07 ms/step.
