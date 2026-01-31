#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  ./run_mc_tmux.sh -m MODEL -P PHASEIDS [-x "region1 region2 ..."] [options]

Required:
  -m, --model MODEL          Agent name (string). Example: Agent_P2-v11.5-multi-SEMIDEF
  -P, --phaseids IDS         PhaseID codes as comma/space separated list.
                             Examples: -P 123,34    OR   -P "123 34"

Optional:
  -x, --regions LIST         Regions to simulate (space-separated string).
                             Default: "aposelene leaving_aposelene approaching_aposelene periselene"
  -s, --seed SEED            Seed (default: 1753110)
  -n, --n-sim N              Number of simulations (default: 100)
  -i, --image IMAGE          Podman image (default: paiton:v01)
  -w, --workdir PATH         Host workdir to cd into before running (default: main/relativeGuidance)
  -l, --log-dir DIR          Log directory (default: tmux_logs)
  --dry-run                  Print commands, do not start tmux sessions
  -h, --help                 Show this help

Notes:
  - Each PhaseID code (e.g. 123, 34) is treated as an independent simulation batch.
  - Session name includes PhaseID + region + model.
EOF
}

# Defaults (can be overridden by env or flags)
IMAGE="${IMAGE:-paiton:v01}"
HOST_WORKDIR="${HOST_WORKDIR:-main/relativeGuidance}"
SEED="${SEED:-1753110}"
N_SIM="${N_SIM:-100}"
REGIONS_DEFAULT=("aposelene" "leaving_aposelene" "approaching_aposelene" "periselene")
LOG_DIR="${LOG_DIR:-tmux_logs}"

MODEL=""
PHASEIDS_RAW=""
REGIONS_RAW=""
DRY_RUN=0

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
	case "$1" in
		-m|--model)
			MODEL="${2:-}"; shift 2;;
		-P|--phaseids)
			PHASEIDS_RAW="${2:-}"; shift 2;;
		-x|--regions)
			REGIONS_RAW="${2:-}"; shift 2;;
		-s|--seed)
			SEED="${2:-}"; shift 2;;
		-n|--n-sim)
			N_SIM="${2:-}"; shift 2;;
		-i|--image)
			IMAGE="${2:-}"; shift 2;;
		-w|--workdir)
			HOST_WORKDIR="${2:-}"; shift 2;;
		-l|--log-dir)
			LOG_DIR="${2:-}"; shift 2;;
		--dry-run)
			DRY_RUN=1; shift;;
		-h|--help)
			usage; exit 0;;
		*)
			echo "Unknown option: $1" >&2
			usage
			exit 2;;
	esac
done

# ---- Validate ----
if [[ -z "$MODEL" ]]; then
	echo "ERROR: missing required --model" >&2
	usage; exit 2
fi

if [[ -z "$PHASEIDS_RAW" ]]; then
	echo "ERROR: missing required --phaseids" >&2
	usage; exit 2
fi

# cd into workdir if it exists
if [[ -n "$HOST_WORKDIR" ]]; then
	if [[ -d "$HOST_WORKDIR" ]]; then
		cd "$HOST_WORKDIR"
	else
		echo "ERROR: workdir not found: $HOST_WORKDIR" >&2
		exit 2
	fi
fi

mkdir -p "$LOG_DIR"

# Preflight: ensure the image is resolvable/present
if ! podman image exists "$IMAGE"; then
	echo "Podman image '$IMAGE' not found or unqualified short name not resolvable." >&2
	echo "Use a fully-qualified name (e.g., 'localhost/paiton:v01' or 'docker.io/...') or pull the image first." >&2
	exit 1
fi

# ---- Parse phaseIDs: allow "123,34" or "123 34" ----
# convert commas to spaces, squeeze spaces
PHASEIDS_RAW="${PHASEIDS_RAW//,/ }"
read -r -a PHASEIDS <<<"$PHASEIDS_RAW"

# ---- Parse regions ----
if [[ -n "$REGIONS_RAW" ]]; then
	read -r -a REGIONS <<<"$REGIONS_RAW"
else
	REGIONS=("${REGIONS_DEFAULT[@]}")
fi

# ---- Run ----
for phaseid in "${PHASEIDS[@]}"; do
	# quick sanity: ensure only digits (adjust if your IDs can be alphanumeric)
	if [[ ! "$phaseid" =~ ^[0-9]+$ ]]; then
		echo "ERROR: phaseid '$phaseid' is not numeric. If you need alphanumerics, relax the regex." >&2
		exit 2
	fi

	for region in "${REGIONS[@]}"; do
		# session name: keep it tmux-friendly
		safe_model="${MODEL// /_}"
		session="MC_ID${phaseid}_${region}_${safe_model}"
		log="$LOG_DIR/${session}.log"

		echo "Starting session $session (phaseID=$phaseid, model=$MODEL, region=$region)..."

		if tmux has-session -t "$session" 2>/dev/null; then
			echo "Session $session already exists. Killing it to start a new one."
			if [[ $DRY_RUN -eq 0 ]]; then
				tmux kill-session -t "$session"
			fi
		fi

		cmd=$(
			cat <<EOF
set -euo pipefail
podman run --rm -it --entrypoint "" -v "$(pwd)":/code -w /code "$IMAGE" \\
	python3 MonteCarlo_eval.py -p "$phaseid" -m "$MODEL" -s "$SEED" -n "$N_SIM" -r False -x "$region" -y \\
	|& tee "$log"
status=\${PIPESTATUS[0]}
echo "exit code: \$status" | tee -a "$log"
exit "\$status"
EOF
		)

		if [[ $DRY_RUN -eq 1 ]]; then
			echo "DRY RUN: tmux new-session -d -s \"$session\" bash -lc '...'"
			echo "--------- command ---------"
			echo "$cmd"
			echo "---------------------------"
			continue
		fi

		if tmux new-session -d -s "$session" bash -lc "$cmd"; then
			echo "Session started (log: $log)"
		else
			echo "Failed to start session $session" >&2
			continue
		fi
	done
done

echo "Sessions started. Attach with: tmux attach -t SESSION_NAME"
echo "Logs under: $LOG_DIR/*.log"
