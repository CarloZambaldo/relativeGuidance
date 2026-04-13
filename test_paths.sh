#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Usage:
#   ./run_mc_tmux.sh -p 2 -q 5 -m "Agent_P2-v11.5-multi-SEMIDEF"
#
# Args:
#   -p | --phase           phaseID (integer)
#   -q | --noise           noisePercentage (number, e.g. 0, 0.005, 0.01, ...)
#   -m | --model           agentModel (string)
#
# What it does:
#   For ALL phases in PHASES[], it:
#     - creates a tmux session
#     - inside tmux runs a podman container
#     - runs: python3 MonteCarlo_eval.py ... with phaseID, noisePercentage, agentModel
#     - tees logs under LOG_DIR/
# -----------------------------------------------------------------------------

# -------------------------
# Config (edit if needed)
# -------------------------

IMAGE="${IMAGE:-paiton:v01}"
HOST_WORKDIR="main/relativeGuidance"   # adjust if needed
SEED="${SEED:-1753110}"
N_SIM="${N_SIM:-1}"

P_VALUES=(1 2)              # phase values to simulate
MODELS=("_NO_AGENT_" "_NO_AGENT_")  # models to use (index p-1)
NOISE_VAL="${NOISE_VAL:-0.0}"

PHASES=("aposelene")

LOG_DIR="tmux_logs"
mkdir -p "$LOG_DIR"

PHASE_ID=""
NOISE_PCT=""
AGENT_MODEL=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		-p|--phase)
			PHASE_ID="${2:-}"; shift 2;;
		-e|--noise)
			NOISE_PCT="${2:-}"; shift 2;;
		-m|--model)
			AGENT_MODEL="${2:-}"; shift 2;;
		-h|--help)
			usage; exit 0;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 2;;
	esac
done

if [[ -z "$PHASE_ID" || -z "$NOISE_PCT" || -z "$AGENT_MODEL" ]]; then
	echo "Error: missing required arguments." >&2
	usage
	exit 2
fi

# Basic validation (lightweight, no dependencies)
if ! [[ "$PHASE_ID" =~ ^[0-9]+$ ]]; then
	echo "Error: phaseID must be an integer, got: '$PHASE_ID'" >&2
	exit 2
fi
if ! [[ "$NOISE_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "Error: noisePercentage must be a number, got: '$NOISE_PCT'" >&2
	exit 2
fi

# Preflight: ensure the image is resolvable/present
if ! podman image exists "$IMAGE"; then
	echo "Podman image '$IMAGE' not found or unqualified short name not resolvable." >&2
	echo "Use a fully-qualified name (e.g., 'localhost/paiton:v01' or 'docker.io/...') or pull the image first." >&2
	exit 1
fi


# -------------------------
# Start one tmux session per phase in PHASES[]
# -------------------------
for region in "${PHASES[@]}"; do
	# sanitize model for filename/session name
	model_tag="$(echo "$AGENT_MODEL" | tr ' /:' '___' | tr -cd '[:alnum:]_.-')"
	session="MC_P${PHASE_ID}_N${NOISE_PCT}_${region}_${model_tag}"
	log="$LOG_DIR/${session}.log"

	echo "Starting session $session (phaseID=$PHASE_ID, noise=$NOISE_PCT, model=$AGENT_MODEL, region=$region)..."

	# Kill existing session to allow reruns
	if tmux has-session -t "$session" 2>/dev/null; then
		echo "Session $session already exists. Killing it to start a new one."
		tmux kill-session -t "$session"
	fi

	# Note: --entrypoint "" keeps your container default shell behavior similar to your original script
	# The python args below assume:
	#   -p = phaseID
	#   -m = agentModel
	#   -s = seed
	#   -n = number of sims
	#   -x = phase name/region
	# Add your noise flag here (example: -q). If your script uses a different flag, change it.
	
	if tmux new-session -d -s "$session" bash -lc "
		set -euo pipefail
		podman run --rm -it --entrypoint \"\" \
			-v /home/czambaldo/main/relativeGuidance/:/code \
			-v /scratch/czambaldo/data:/data -w /code \"$IMAGE\"\
			python3 MonteCarlo_eval.py \
				-p \"$PHASE_ID\" \
				-m \"$AGENT_MODEL\" \
				-e \"$NOISE_PCT\" \
				-s \"$SEED\" \
				-n \"$N_SIM\" \
				-r False \
				-x \"$region\" \
				-y \
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

echo
echo "Sessions started. Attach to one with: tmux attach -t SESSION_NAME"
echo "Logs under: $LOG_DIR/*.log"
echo "List sessions: tmux ls"