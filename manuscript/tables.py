"""Typeset the manuscript tables from the released Windows summary files."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "replication/summary"


def read(name):
    with (SUMMARY/name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(x):
    if x < 1e5:
        return f"{x:,.2f}"
    mantissa, exponent = f"{x:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


risk = {(row["dataset"],int(row["n"]),row["estimator"]):float(row["mean"]) for row in read("risk.csv")}
datasets = ("YearPredictionMSD","Superconductivity")
grids = [sorted({n for d,n,_ in risk if d==dataset}) for dataset in datasets]
rows = []
for ns in zip(*grids, strict=True):
    fields = []
    for d,n in zip(datasets,ns):
        fields.extend([str(n),*(number(risk[d,n,model]) for model in ("min_norm","ridge","pcr"))])
    rows.append(" & ".join(fields)+r" \\")
(HERE/"risk_rows.tex").write_text("\n".join(rows)+"\n")

rows = []
sweep = read("sensitivity.csv")
for c in sorted({float(row["relative_cutoff"]) for row in sweep}):
    subset = [row for row in sweep if float(row["relative_cutoff"])==c]
    mean = max(subset,key=lambda r:float(r["mean"]))
    median = max(subset,key=lambda r:float(r["median"]))
    label = "Default" if c == 0 else rf"$10^{{{int(f'{c:e}'.split('e')[1])}}}$"
    fields = [label,str(sum(int(r["changed_rank"]) for r in subset)),mean["n"],
              number(float(mean["mean"])),median["n"],rf"{100*float(mean['largest_share']):.2f}\%"]
    rows.append(" & ".join(fields)+r" \\")
(HERE/"cutoff_rows.tex").write_text("\n".join(rows)+"\n")
