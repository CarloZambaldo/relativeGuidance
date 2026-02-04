#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_mc_tmux.sh -p <phaseID> -q <noisePercentage> -m <agentModel>
#
# Examples:
#   ./run_mc_tmux.sh -p 2 -q 5 -m _NO_AGENT_
#   ./run_mc_tmux.sh -p 1 -q 0 -m Agent_P2-v11.5-multi-SEMIDEF

# ---- Config ----
IMAGE="${IMAGE:-paiton:v01}"
HOST_WORKDIR="main/relativeGuidance"   # adjust if needed
SEED="${SEED:-1753110}"
N_SIM="${N_SIM:-100}"

LOG_DIR="tmux_logs"
mkdir -p "$LOG_DIR"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_mc_tmux.sh -p <phaseID> -q <noisePercentage> -m <agentModel> [-s <seed>] [-n <nSim>]

Required:
  -p    phaseID (integer)
  -q    noisePercentage (numeric, e.g. 0, 5, 2.5)
  -m    agentModel (string)

Optional:
  -s    seed (default: from $SEED env or script default)
  -n    number of simulations (default: from $N_SIM env or script default)

Environment overrides:
  IMAGE, HOST_WORKDIR, SEED, N_SIM

EOF
}

# ---- Parse flags ----
PHASE_ID=""
NOISE_PERC=""
AGENT_MODEL=""

while getopts ":p:q:m:s:n:h" opt; do
  case "$opt" in
    p) PHASE_ID="$OPTARG" ;;
    q) NOISE_PERC="$OPTARG" ;;
    m) AGENT_MODEL="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    n) N_SIM="$OPTARG" ;;
    h) usage; exit 0 ;;
    :)
      echo "Error: -$OPTARG requires an argument." >&2
      usage
      exit 2
      ;;
    \?)
      echo "Error: unknown option: -$OPTARG" >&2
      usage
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

# ---- Validate required flags ----
if [[ -z "$PHASE_ID" || -z "$NOISE_PERC" || -z "$AGENT_MODEL" ]]; then
  echo "Error: missing required flags." >&2
  usage
  exit 2
fi

if ! [[ "$PHASE_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: -p phaseID must be an integer. Got: '$PHASE_ID'" >&2
  exit 2
fi
if ! [[ "$NOISE_PERC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: -q noisePercentage must be numeric (e.g., 0, 5, 2.5). Got: '$NOISE_PERC'" >&2
  exit 2
fi
if ! [[ "$N_SIM" =~ ^[0-9]+$ ]]; then
  echo "Error: -n N_SIM must be an integer. Got: '$N_SIM'" >&2
  exit 2
fi

# ---- Preflight ----
if ! podman image exists "$IMAGE"; then
  echo "Podman image '$IMAGE' not found or unqualified short name not resolvable." >&2
  echo "Use a fully-qualified name (e.g., 'localhost/paiton:v01' or 'docker.io/...') or pull the image first." >&2
  exit 1
fi

if [[ ! -d "$HOST_WORKDIR" ]]; then
  echo "Warning: HOST_WORKDIR '$HOST_WORKDIR' not found from current dir: $(pwd)" >&2
  echo "         Continuing anyway (adjust HOST_WORKDIR if needed)." >&2
fi

# ---- Build names ----
AGENT_SAFE="$(echo "$AGENT_MODEL" | tr -c '[:alnum:]._+-' '_' )"
NOISE_SAFE="$(echo "$NOISE_PERC" | tr '.' 'p')"

SESSION="MC_P${PHASE_ID}_N${NOISE_SAFE}_${AGENT_SAFE}"
LOG="$LOG_DIR/${SESSION}.log"

echo "Starting session $SESSION (phaseID=$PHASE_ID, noise=$NOISE_PERC, model=$AGENT_MODEL, seed=$SEED, nSim=$N_SIM)..."
echo "Log: $LOG"

# Kill existing session with same name to allow reruns
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Killing it to start a new one."
  tmux kill-session -t "$SESSION"
fi

# ---- Start tmux session ----
tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail

  # run from your project folder (host), mounted into /code
  cd \"$HOST_WORKDIR\" 2>/dev/null || true

  podman run --rm -it --entrypoint \"\" \
    -v \"$(pwd | sed 's:/*$::')\":/code \
    -w /code \
    \"$IMAGE\" \
    python3 MonteCarlo_eval.py \
      -p \"$PHASE_ID\" \
      -q \"$NOISE_PERC\" \
      -m \"$AGENT_MODEL\" \
      -s \"$SEED\" \
      -n \"$N_SIM\" \
    |& tee \"$LOG\"

  status=\${PIPESTATUS[0]}
  echo \"exit code: \$status\" | tee -a \"$LOG\"
  exit \"\$status\"
"

echo "Session started. Attach with:"
echo "  tmux attach -t \"$SESSION\""
echo "Tail log with:"
echo "  tail -f \"$LOG\""
