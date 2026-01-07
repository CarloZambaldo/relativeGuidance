#!/usr/bin/env bash
set -euo pipefail

# Config
IMAGE="${IMAGE:-paiton:v01}"
HOST_WORKDIR="main/relativeGuidance"   # adjust if needed
SEED="${SEED:-1753110}"
N_SIM="${N_SIM:-100}"

P_VALUES=(1 2)              # phase values to simulate
MODELS=("Agent_P1-v11-thesis" "Agent_P2-v11.5-multi-SEMIDEF")  # models to use (index p-1)
PHASES=("aposelene" "leaving_aposelene" "approaching_aposelene" "periselene")

LOG_DIR="tmux_logs"
mkdir -p "$LOG_DIR"

# Preflight: ensure the image is resolvable/present
if ! podman image exists "$IMAGE"; then
  echo "Podman image '$IMAGE' not found or unqualified short name not resolvable." >&2
  echo "Use a fully-qualified name (e.g., 'localhost/paiton:v01' or 'docker.io/...') or pull the image first." >&2
  exit 1
fi

# Start tmux sessions for each combination of parameters
for p in "${P_VALUES[@]}"; do
  model="${MODELS[$((p-1))]}"
  for region in "${PHASES[@]}"; do
    session="MC_P${p}_${region}_${model}"
    log="$LOG_DIR/${session}.log"
    
    echo "Starting session $session (p=$p, m=$model, region=$region)..."
    # Kill existing session with the same name to avoid tmux errors on reruns
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Session $session already exists. Killing it to start a new one."
      tmux kill-session -t "$session"
    fi
    if tmux new-session -d -s "$session" bash -lc "
      set -euo pipefail
      podman run --rm -it --entrypoint \"\" -v \"$(pwd)\":/code -w /code \"$IMAGE\" \
        python3 MonteCarlo_eval.py -p \"$p\" -m \"$model\" -s \"$SEED\" -n \"$N_SIM\" -r False -x \"$region\" -y \
        |& tee \"$log\"
      status=\${PIPESTATUS[0]}
      echo \"exit code: \$status\" | tee -a \"$log\"
      exit \"\$status\"
    "; then
      echo "Session started (log: $log)"
    else
      echo "Failed to start session $session" >&2
      continue
    fi
  done
done

echo "Sessions started. Attach to one with: tmux attach -t SESSION_NAME"
echo "Logs under: $LOG_DIR/*.log"


# podman run --rm -it -v \"$(pwd)\":/code \"$IMAGE\" \
#        bash -lc "python3 MonteCarlo_eval.py -p \"$p\" -m \"$model\" -s \"$SEED\" -n \"$N_SIM\" -r False -x \"$region\" -y" \
#      |& tee \"$log\"
