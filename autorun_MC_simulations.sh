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

LOG_DIR="tmux_logs"
mkdir -p "$LOG_DIR"

for p in "${P_VALUES[@]}"; do
  model="${MODELS[$((p-1))]}"
  for region in "${PHASES[@]}"; do
    session="MC_P${p}_${region}"
    log="$LOG_DIR/${session}.log"
    echo "Starting session $session (p=$p, m=$model, region=$region)..."
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
    tmux new-session -d -s "$session" bash -lc "
      cd \"$(pwd)\" && podman run --rm -v \"$(pwd)\":/code \"$IMAGE\" \
        bash -lc 'cd /code && python3 MonteCarlo_eval.py -p \"$p\" -m \"$model\" -s \"$SEED\" -n \"$N_SIM\" -r False -x \"$region\" -y' \
      |& tee \"$log\"; echo \"exit code: \$?\" | tee -a \"$log\""
  done
done

echo "Logs in: $LOG_DIR/*.log"
echo "Sessions started. Attach to one with: tmux attach -t SESSION_NAME"
echo "List sessions with: tmux ls"