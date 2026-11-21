"""Check archived spectra against the shared losses and summarize peak influence."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t

from replication.summarize import CUTS, GRIDS, HERE, arrange, col, read


def diagnose(root):
    risk = arrange(read(root/"losses.csv"), lambda r: (r["dataset"],int(r["n"])))
    sweep = arrange(read(root/"cutoff.csv"), lambda r: (float(r["relative_cutoff"]),int(r["n"])))
    windows = {"YearPredictionMSD": (81,86,88,90,92,95,99),
               "Superconductivity": (41,61,73,77,79,81,83,85,89)}
    near = set()
    for d,window in windows.items():
        spectra = arrange(read(root/f"spectra_{d}.csv"), lambda r: int(r["sample_size"]))
        if set(spectra) != set(window):
            raise ValueError("incomplete spectral window")
        ratios = []
        for n,rows in spectra.items():
            if not np.array_equal(col(rows,"archived_mse"), col(risk[d,n],"min_norm")):
                raise ValueError(f"spectra/loss mismatch at {d}, {n}")
            ratio = col(rows,"s_min")/col(rows,"s1")
            ratios.extend(ratio)
            if d=="Superconductivity":
                near.update((n,int(r)) for r in np.flatnonzero(ratio<1e-10))
        print(f"{d}: {len(ratios)} archived spectra; smallest retained s/s1 = {min(ratios):.6g}")
    cert = {(n,r):row for (c,n),rows in sweep.items() if c==0
            for r,row in enumerate(rows) if row["field_rank_lower"]}
    if set(cert)!=near or any(row["field_rank_lower"]!=row["rank_upper"] for row in cert.values()):
        raise ValueError("rank certificate selection or bounds disagree with recorded spectra")
    print("Certified minus default ranks:", sorted(Counter(int(row["rank_upper"])-int(row["rank"]) for row in cert.values()).items()))
    grid = np.array(GRIDS["Superconductivity"][1])
    for c in CUTS:
        matrix = np.array([col(sweep[c,n],"mse") for n in grid])
        # Recompute each sum directly. Subtracting the largest term from a
        # large aggregate can lose digits in the remaining tail.
        peaks = [int(grid[np.argmax(matrix[:,np.arange(100)!=r].mean(axis=1))]) for r in range(100)]
        delta = col(sweep[c,73],"mse")-col(sweep[c,83],"mse")
        half = t.ppf(.975,99)*delta.std(ddof=1)/10
        print(f"floor {c:g}: leave-one-permutation-out mean peaks {sorted(Counter(peaks).items())}; "
              f"paired mean MSE at 73 minus 83 = {delta.mean():.5g} "
              f"[{delta.mean()-half:.5g}, {delta.mean()+half:.5g}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=HERE)
    diagnose(parser.parse_args().inputs)
