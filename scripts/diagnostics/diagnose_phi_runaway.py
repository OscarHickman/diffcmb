"""Per sky of an ensemble: ln(C_L^phiphi / truth), bin-averaged, by sixths of the chain, and the
last third's mean over its sd (ROADMAP T1.2a, the A_phi = 1 trap; a runaway climbs by >~ +0.5 in
[30,60) from the first sixth and plateaus, at +0.9 to +3.2 so far; a frozen offset sits flat
at |z| >~ 3).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnostics/diagnose_phi_runaway.py ENSEMBLE_DIR [ENSEMBLE_DIR ...]
"""
import glob
import sys

import numpy as np

BINS = {"[10,30)": (10, 30), "[30,60)": (30, 60), "[60,64)": (60, 64)}
for indir in sys.argv[1:]:
    print("==", indir)
    for f in sorted(glob.glob(f"{indir}/chain_r[0-9][0-9][0-9].npz")):
        d = np.load(f)
        lmax = int(d["lmax"])
        s = d["cl_phiphi_samples"]
        t = np.log(d["cl_phiphi_true"][2:lmax])
        out = []
        for name, (lo, hi) in BINS.items():
            x = s[:, lo - 2:hi - 2].mean(1) - t[lo - 2:hi - 2].mean()
            six = [c.mean() for c in np.array_split(x, 6)]
            last = x[len(x) * 2 // 3:]
            out.append((name, six, last.mean() / last.std()))
        r = f[-7:-4]
        mid = out[1]
        print(f"r{r} [30,60) sixths " + " ".join(f"{v:+.2f}" for v in mid[1]) +
              f" | z_last {mid[2]:+.1f} | [10,30) end {out[0][1][-1]:+.2f}  [60,64) end {out[2][1][-1]:+.2f}")
