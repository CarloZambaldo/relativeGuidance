#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_mc_tmux.sh -p <phaseID> -e <noisePercent> -m <agentModel> [-s <seed>] [-n <nSamples>] [-r <True|False>] [-x <posMode>] [-y]
#
# Examples:
#   ./run_mc_tmux.sh -p 2 -e 0.03 -m "PPO_Agent_Phase2_v02" -s 42 -n 100 -r True -x aposelene -y
#   ./run_mc_tmux.sh -p 1 -e 0 -m "_NO_AGENT_" -n 200 -r False -x periselene -y

# ---------------- Config ----------------
IMAGE="${IMAGE:-paiton:v01}"
HOST_WORKDIR="${HOST_WORKDIR:-main/relativeGuidance}"  # adjust if needed
LOG_DIR="tmux_logs"
mkdir -p "$LOG_DIR"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_mc_tmux.sh -p <phaseID> -e <noisePercent> -m <agentModel> [options]

Required:
  -p    phaseID (int)                      -> MonteCarlo_eval.py: -p / --phase
  -e    navigation noise percent (float)   -> MonteCarlo_eval.py: -e / --navigation-noise-percent (e.g., 0.03 for 3%)
  -m    agent model name (string)          -> MonteCarlo_eval.py: -m / --model

Options (passed through):
  -s    seed (int or "None")               -> -s / --seed
  -n    n-samples (int)                    -> -n / --n-samples
  -r    render ("True" or "False")         -> -r / --render
  -x    position-mode (string)             -> -x / --position-mode
  -y    skip-acknowledge (flag)            -> -y / --skip-acknowledge
  -h    help

Env overrides:
  IMAGE, HOST_WORKDIR

Examples:
  ./run_mc_tmux.sh -p 2 -e 0.03 -m "PPO_Agent_Phase2_v02" -s 42 -n 100 -r True -x aposelene -y
EOF
}

# ---------------- Parse flags ----------------
PHASE_ID=""
NOISE_PERC=""
AGENT_MODEL=""

SEED="None"
N_SAMPLES="1"
RENDER="True"
POS_MODE="aposelene"
SKIP_ACK=false

while getopts ":p:e:m:s:n:r:x:yh" opt; do
  case "$opt" in
    p) PHASE_ID="$OPTARG" ;;
    e) NOISE_PERC="$OPTARG" ;;
    m) AGENT_MODEL="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    n) N_SAMPLES="$OPTARG" ;;
    r) RENDER="$OPTARG" ;;
    x) POS_MODE="$OPTARG" ;;
    y) SKIP_ACK=true ;;
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

# ---------------- Validate required ----------------
if [[ -z "$PHASE_ID" || -z "$NOISE_PERC" || -z "$AGENT_MODEL" ]]; then
  echo "Error: missing required flags (-p, -e, -m)." >&2
  usage
  exit 2
fi

if ! [[ "$PHASE_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: -p phaseID must be an integer. Got: '$PHASE_ID'" >&2
  exit 2
fi
if ! [[ "$NOISE_PERC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: -e noisePercent must be numeric (e.g., 0, 0.03, 2.5). Got: '$NOISE_PERC'" >&2
  exit 2
fi
if ! [[ "$N_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "Error: -n n-samples must be an integer. Got: '$N_SAMPLES'" >&2
  exit 2
fi
# SEED can be "None" or an integer
if [[ "$SEED" != "None" && ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Error: -s seed must be an integer or 'None'. Got: '$SEED'" >&2
  exit 2
fi
# render must be True/False (case-insensitive)
if ! [[ "$RENDER" =~ ^(True|False|true|false)$ ]]; then
  echo "Error: -r render must be True/False. Got: '$RENDER'" >&2
  exit 2
fi

# ---------------- Preflight ----------------
if ! podman image exists "$IMAGE"; then
  echo "Podman image '$IMAGE' not found or unqualified short name not resolvable." >&2
  echo "Use a fully-qualified name (e.g., 'localhost/paiton:v01' or 'docker.io/...') or pull the image first." >&2
  exit 1
fi

if [[ ! -d "$HOST_WORKDIR" ]]; then
  echo "Error: HOST_WORKDIR '$HOST_WORKDIR' not found from current dir: $(pwd)" >&2
  exit 1
fi

# ---------------- Session & log names ----------------
AGENT_SAFE="$(echo "$AGENT_MODEL" | tr -c '[:alnum:]._+-' '_' )"
POS_SAFE="$(echo "$POS_MODE" | tr -c '[:alnum:]._+-' '_' )"
NOISE_SAFE="$(echo "$NOISE_PERC" | tr '.' 'p')"

SESSION="MC_P${PHASE_ID}_N${NOISE_SAFE}_${POS_SAFE}_${AGENT_SAFE}"
LOG="$LOG_DIR/${SESSION}.log"

echo "Starting session $SESSION"
echo "  phaseID:   $PHASE_ID"
echo "  noise:     $NOISE_PERC"
echo "  model:     $AGENT_MODEL"
echo "  seed:      $SEED"
echo "  n-samples: $N_SAMPLES"
echo "  render:    $RENDER"
echo "  pos-mode:  $POS_MODE"
echo "  -y:        $SKIP_ACK"
echo "Log: $LOG"

# Kill existing session with same name to allow reruns
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Killing it to start a new one."
  tmux kill-session -t "$SESSION"
fi

# Build the python command (as an array to preserve quoting)
PY_CMD=(python3 MonteCarlo_eval.py
  -p "$PHASE_ID"
  -m "$AGENT_MODEL"
  -s "$SEED"
  -n "$N_SAMPLES"
  -r "$RENDER"
  -x "$POS_MODE"
  -e "$NOISE_PERC"
)

if $SKIP_ACK; then
  PY_CMD+=(-y)
fi

# ---------------- Start tmux session ----------------
tmux new-session -d -s "$SESSION" bash -lc "
  set -euo pipefail
  cd \"$HOST_WORKDIR\"

  podman run --rm -it --entrypoint \"\" \
    -v \"$(pwd)\":/code \
    -w /code \
    \"$IMAGE\" \
    \"\${PY_CMD[@]}\" \
    |& tee \"$LOG\"

  status=\${PIPESTATUS[0]}
  echo \"exit code: \$status\" | tee -a \"$LOG\"
  exit \"\$status\"
"

echo "Session started. Attach with:"
echo "  tmux attach -t \"$SESSION\""
echo "Tail log with:"
echo "  tail -f \"$LOG\""
