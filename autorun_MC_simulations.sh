#!/usr/bin/env bash
set -euo pipefail

SESSION="MC_simulations"
IMAGE="paiton:v01"
HOST_WORKDIR="main/relativeGuidance"   # adjust if needed
SEED=1753110
N_SIM=100

P_VALUES=(1 2)              # phase values to simulate
MODELS=("Agent_P1-v11-thesis" "Agent_P2-v11.5-multi-SEMIDEF")          # models to use
PHASES=("aposelene" "leaving_aposelene" "approaching_aposelene" "periselene")

# Start tmux sessions for each combination of parameters
for p in "${P_VALUES[@]}"; do
  model="${MODELS[$((p-1))]}"
  for region in "${PHASES[@]}"; do
    session="MC_P${p}_${region}"
    echo "Starting session $session (p=$p, m=$model, region=$region)..."
    tmux new-session -d -s "$session" "podman run --rm -it -v \"$(pwd)\":/code/ \"$IMAGE\" bash -lc 'cd /code && python3 MonteCarlo_eval.py -p \"$p\" -m \"$model\" -s \"$SEED\" -n \"$N_SIM\" -r False -x \"$region\" -y'"
    echo "Session started."
  done
done

echo "Sessions started. Attach to one with: tmux attach -t SESSION_NAME"
