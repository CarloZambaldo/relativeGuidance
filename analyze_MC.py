"""
Aggregate MonteCarlo_eval .mat results into a compact JSON of per-simulation
scalars (success/fail/out-of-time, DeltaV, TOF, mean GNC execution time,
terminal miss and contact velocity, agent statistics).

Usage:
    python3 analyze_MC.py <dir_with_mat_files> [output.json]

Designed to run on the MAGI servers (inside the paiton container) where the
full 1.7 GB .mat histories live; the JSON output is small enough to download
and use locally for plotting and paper tables.
"""
import sys
import re
import json
from pathlib import Path

import numpy as np
import scipy.io

FNAME_RE = re.compile(
    r"^MC_P(?P<phase>\d)_N(?P<noise>[0-9.]+)_(?P<region>.+?)__(?P<model>.+)_"
    r"(?P<stamp>\d{4}_\d{2}_\d{2}_at_\d{2}_\d{2})\.mat$"
)


def analyze_file(path):
    m = FNAME_RE.match(path.name)
    if not m:
        print(f"  ! skipping (unparsable name): {path.name}")
        return None
    meta = m.groupdict()

    d = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)["data"]
    param = d.param
    xc, tc = float(param.xc), float(param.tc)
    freqGNC = float(param.freqGNC)          # adimensional (Hz * tc)
    dt_adim = 1.0 / freqGNC                 # adimensional GNC step
    dt_s = dt_adim * tc                     # [s] (= 0.2 s at 5 Hz)

    success = np.asarray(d.success, dtype=bool)
    fail = np.asarray(d.fail, dtype=bool)
    oot = ~(success | fail)
    n = success.size

    term_idx = np.asarray(d.terminalTimeIndex, dtype=int)
    u = np.asarray(d.controlAction)         # (T, 3, N) adimensional acceleration
    cpu = np.asarray(d.CPUExecTimeHistory)  # (T-1, N) [s]
    terminal = np.asarray(d.terminalState)  # (6, N) adimensional LVLH
    agent_act = np.asarray(d.AgentAction)   # (T-1, N)
    obot = np.asarray(d.OBoTUsage)          # (T-1, N)
    true_rel = np.asarray(d.trueRelativeStateHistory_L)  # (T-1, 6, N)

    # aim point (adimensional LVLH): holding state for P1, docking port for P2
    aim = np.array([0.0, -4.0 / xc, 0.0]) if meta["phase"] == "1" else np.zeros(3)

    acc2ms = xc * 1e3 / tc                  # adim velocity -> m/s

    dv = np.zeros(n)
    texec_ms = np.zeros(n)
    n_recompute = np.zeros(n, dtype=int)
    obot_frac = np.zeros(n)
    final_err_m = np.zeros(n)
    min_err_m = np.zeros(n)
    for i in range(n):
        k = term_idx[i] if term_idx[i] > 0 else u.shape[0] - 1
        dv[i] = np.sum(np.linalg.norm(u[: k + 1, :, i], axis=1)) * dt_adim * acc2ms
        steps = cpu[:k, i]
        texec_ms[i] = 1e3 * (steps[steps > 0].mean() if np.any(steps > 0) else 0.0)
        n_recompute[i] = int(np.sum(agent_act[:k, i] == 1))
        obot_frac[i] = float(np.mean(obot[:k, i])) if k > 0 else 0.0
        kk = min(k, true_rel.shape[0] - 1)
        err = np.linalg.norm(true_rel[: kk + 1, :3, i] - aim, axis=1) * xc * 1e3  # [m]
        final_err_m[i] = err[-1]
        min_err_m[i] = err.min()

    tof_min = term_idx * dt_s / 60.0

    out = {
        **meta,
        "n": int(n),
        "success": success.astype(int).tolist(),
        "fail": fail.astype(int).tolist(),
        "oot": oot.astype(int).tolist(),
        "dv_ms": np.round(dv, 4).tolist(),
        "tof_min": np.round(tof_min, 3).tolist(),
        "texec_ms": np.round(texec_ms, 4).tolist(),
        "n_recompute": n_recompute.tolist(),
        "obot_frac": np.round(obot_frac, 4).tolist(),
        "final_err_m": np.round(final_err_m, 3).tolist(),
        "min_err_m": np.round(min_err_m, 3).tolist(),
        # terminal state: positions in cm, velocities in cm/s (LVLH: R, V, H)
        "final_R_cm": np.round(terminal[0] * xc * 1e5, 3).tolist(),
        "final_V_cm": np.round(terminal[1] * xc * 1e5, 3).tolist(),
        "final_H_cm": np.round(terminal[2] * xc * 1e5, 3).tolist(),
        "final_vR_cms": np.round(terminal[3] * xc / tc * 1e5, 4).tolist(),
        "final_vV_cms": np.round(terminal[4] * xc / tc * 1e5, 4).tolist(),
        "final_vH_cms": np.round(terminal[5] * xc / tc * 1e5, 4).tolist(),
    }

    sr = 100.0 * success.mean()
    dv_ok = dv[success] if success.any() else dv
    print(f"  {path.name}: success {sr:.0f}%  dV(succ) {dv_ok.mean():.2f}±{dv_ok.std():.2f} m/s")
    return out


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "./Simulations")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else src / "MC_summary.json")

    results = []
    files = sorted(src.glob("MC_*.mat"))
    print(f"Aggregating {len(files)} files from {src} ...")
    for f in files:
        try:
            r = analyze_file(f)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ! ERROR on {f.name}: {e}")

    with open(dst, "w") as fh:
        json.dump(results, fh)
    print(f"Wrote {dst} ({dst.stat().st_size/1e6:.2f} MB, {len(results)} runs)")


if __name__ == "__main__":
    main()
