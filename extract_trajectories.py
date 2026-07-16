"""
Extract a handful of full example trajectories (position/velocity/control/
agent-action/OBoT-usage time histories) from one or more MonteCarlo_eval .mat
result files, for explanatory plots (3D relative dynamics + constraint cone,
time histories) analogous to matlabScripts/MonteCarloPlots.m figure 1/2.

The full .mat files are large (~1.7 GB, one per Monte Carlo run of 100 sims);
this script picks a few representative simulation indices and writes a small
JSON with only their trajectories, meant to be downloaded and plotted locally.

Usage:
    python3 extract_trajectories.py <mat_file> [output.json] [--n N] [--select best,median,worst,first]
"""
import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import scipy.io

FNAME_RE = re.compile(
    r"^MC_P(?P<phase>\d)_N(?P<noise>[0-9.]+)_(?P<region>.+?)__(?P<model>.+)_\d{4}_\d{2}_\d{2}_at_\d{2}_\d{2}\.mat$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_file")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=6, help="number of example trajectories to extract")
    ap.add_argument("--stride", type=int, default=1, help="keep 1 every `stride` GNC steps (decimation, for compact output on long flights)")
    args = ap.parse_args()

    mat_path = Path(args.mat_file)
    out_path = Path(args.output) if args.output else mat_path.with_suffix("").with_name(mat_path.stem + "_traj.json")

    d = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["data"]
    param = d.param
    xc, tc = float(param.xc), float(param.tc)
    freqGNC = float(param.freqGNC)
    dt_s = tc / freqGNC

    success = np.asarray(d.success, dtype=bool)
    fail = np.asarray(d.fail, dtype=bool)
    term_idx = np.asarray(d.terminalTimeIndex, dtype=int)
    true_rel = np.asarray(d.trueRelativeStateHistory_L)   # (T-1, 6, N) adim
    u = np.asarray(d.controlAction)                       # (T, 3, N) adim
    agent_act = np.asarray(d.AgentAction)                 # (T-1, N)
    obot = np.asarray(d.OBoTUsage)                         # (T-1, N)
    n = success.size

    # dV per sim (for ranking best/median/worst among successful runs)
    acc2ms = xc * 1e3 / tc
    dv = np.zeros(n)
    for i in range(n):
        k = term_idx[i] if term_idx[i] > 0 else u.shape[0] - 1
        dv[i] = np.sum(np.linalg.norm(u[: k + 1, :, i], axis=1)) * (1.0 / freqGNC) * acc2ms

    succ_idx = np.where(success)[0]
    fail_idx = np.where(fail)[0]
    oot_idx = np.where(~(success | fail))[0]

    chosen = []
    if succ_idx.size:
        order = succ_idx[np.argsort(dv[succ_idx])]
        chosen.append(("best_dV", order[0]))
        chosen.append(("median_dV", order[len(order) // 2]))
        chosen.append(("worst_dV_success", order[-1]))
    if fail_idx.size:
        chosen.append(("example_crash", fail_idx[0]))
    if oot_idx.size:
        chosen.append(("example_out_of_time", oot_idx[0]))
    # pad with plain sequential examples up to n
    i = 0
    while len(chosen) < args.n and i < n:
        if i not in [c[1] for c in chosen]:
            chosen.append((f"sim_{i}", i))
        i += 1
    chosen = chosen[: args.n]

    m = FNAME_RE.match(mat_path.name)
    meta = m.groupdict() if m else {}

    out = {
        "file": mat_path.name,
        "phase": str(d.phaseID),
        **meta,
        "param": {"xc": xc, "tc": tc, "dt_s": dt_s},
        "trajectories": [],
    }
    for label, idx in chosen:
        k = term_idx[idx] if term_idx[idx] > 0 else true_rel.shape[0] - 1
        k = min(k, true_rel.shape[0] - 1)
        pos_km = true_rel[: k + 1, :3, idx] * xc          # (k+1, 3) km
        vel_ms = true_rel[: k + 1, 3:6, idx] * xc * 1e3 / tc  # m/s
        ctrl = u[: k + 1, :, idx] * (xc * 1e3 / tc**2)        # m/s^2
        time_s = np.arange(k + 1) * dt_s
        stride = max(1, args.stride)
        sel = np.arange(0, k + 1, stride)
        if sel[-1] != k:  # always keep the exact terminal point
            sel = np.append(sel, k)
        aa = agent_act[:k, idx]
        ob = obot[:k, idx]
        sel_aa = sel[sel < aa.shape[0]]
        out["trajectories"].append({
            "label": label,
            "sim_id": int(idx),
            "success": bool(success[idx]),
            "fail": bool(fail[idx]),
            "dv_ms": round(float(dv[idx]), 3),
            "time_s": np.round(time_s[sel], 2).tolist(),
            "pos_km": np.round(pos_km[sel], 5).tolist(),
            "vel_ms": np.round(vel_ms[sel], 5).tolist(),
            "ctrl_ms2": np.round(ctrl[sel], 7).tolist(),
            "agent_action": aa[sel_aa].astype(int).tolist(),
            "obot_usage": ob[sel_aa].astype(bool).astype(int).tolist(),
        })

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {out_path} ({out_path.stat().st_size/1e3:.1f} KB, {len(out['trajectories'])} trajectories)")


if __name__ == "__main__":
    main()
