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

### TODO / next steps for the paper
- Pull the 80 `.mat` files from casper and run the aggregate analysis (ΔV, miss distance, trajectories) locally.
- Decide how to present P1 high-noise degradation (agent ≈ safe at ≥2%) and P2 safe-mode advantage at high noise.
