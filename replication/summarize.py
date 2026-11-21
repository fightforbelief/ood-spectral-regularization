"""Recompute risk summaries, paired intervals, sensitivity tables and figures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
GRIDS = {"YearPredictionMSD": (90, (45,68,81,86,88,90,92,95,99,113,135,180,270,450,900)),
         "Superconductivity": (81, (41,61,73,77,79,81,83,85,89,101,122,162,243,405,810))}
CUTS = (0.,1e-12,1e-10,1e-8,1e-6)


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def arrange(rows, keys):
    groups = {}
    for row in rows:
        k = keys(row)
        by_rep = groups.setdefault(k, {})
        r = int(row["repetition"])
        if r in by_rep:
            raise ValueError(f"Repeated repetition {r} in {k}")
        by_rep[r] = row
    for k, by_rep in groups.items():
        if set(by_rep) != set(range(100)):
            raise ValueError(f"Missing repetitions for {k}")
        groups[k] = [by_rep[r] for r in range(100)]
    return groups


def col(rows, name):
    x = np.array([float(row[name]) for row in rows])
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite {name}")
    return x


def decorate(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def summarize(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    pairs = arrange(read(inputs/"losses.csv"), lambda r: (r["dataset"], int(r["n"])))
    sweep = arrange(read(inputs/"cutoff.csv"), lambda r: (float(r["relative_cutoff"]), int(r["n"])))
    if set(pairs) != {(d,n) for d,(_,ns) in GRIDS.items() for n in ns}:
        raise ValueError("unexpected benchmark grid")
    super_grid = GRIDS["Superconductivity"][1]
    if set(sweep) != {(c,n) for c in CUTS for n in super_grid}:
        raise ValueError("incomplete cutoff grid")
    primary, contrasts, sensitivities = [], [], []
    for (d,n), rows in sorted(pairs.items()):
        for model in ("min_norm","ridge","pcr","null"):
            x = col(rows, model)
            primary.append(dict(dataset=d,n=n,estimator=model,mean=x.mean(),median=np.median(x),
                                largest_share=x.max()/x.sum()))
        for a,b in (("pcr","ridge"),("min_norm","ridge"),("pcr","min_norm"),("null","ridge")):
            diff = col(rows,a)-col(rows,b)
            half = t.ppf(.975,99)*diff.std(ddof=1)/10.
            contrasts.append(dict(dataset=d,n=n,a=a,b=b,mean=diff.mean(),low=diff.mean()-half,
                                  high=diff.mean()+half,b_lower=int((diff>0).sum())))
    for (c,n), rows in sorted(sweep.items()):
        x = col(rows,"mse")
        original = sweep[0., n]
        archived = col(pairs["Superconductivity",n], "min_norm")
        if not np.allclose(col(original,"mse"), archived, rtol=1e-5, atol=0):
            raise ValueError(f"primary and sensitivity runs disagree at {n}")
        sensitivities.append(dict(relative_cutoff=c,n=n,mean=x.mean(),median=np.median(x),
                                  changed_rank=int((col(rows,"rank")!=col(original,"rank")).sum()),
                                  largest_share=x.max()/x.sum()))
    save(output/"risk.csv",primary)
    save(output/"paired.csv",contrasts)
    save(output/"sensitivity.csv",sensitivities)
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.labelsize":9})
    fig,axes=plt.subplots(1,2,figsize=(7,2.75),layout="constrained")
    for ax,(d,(p,grid)) in zip(axes,GRIDS.items()):
        for method,label,style,color in [("min_norm","Minimum norm","-o","#852e36"),
                                         ("ridge","Ridge","-s","#096b72"),("pcr","PCR","--^","#544099"),
                                         ("null","Training mean",":","#494949")]:
            v=[next(z["mean"] for z in primary if (z["dataset"],z["n"],z["estimator"])==(d,n,method)) for n in grid]
            ax.plot(np.array(grid)/p,v,style,ms=3,lw=1.3,color=color,label=label)
        ax.axvline((p+1)/p,color=".5",lw=.8,ls="--")
        ax.set(xscale="log",yscale="log",xlabel="Training size / number of predictors",title=d)
        decorate(ax)
    axes[0].set_ylabel("Mean holdout MSE")
    axes[0].legend(fontsize=7,frameon=False)
    fig.savefig(output/"curves.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout="constrained")
    for c,label,style,color in zip(CUTS,["Default",r"$10^{-12}$",r"$10^{-10}$",r"$10^{-8}$",r"$10^{-6}$"],
                                   ["-o","--s","-.^",":D","-x"],["#24242a","#ab384d","#835bac","#057b72","#996719"]):
        vals=[z for z in sensitivities if z["relative_cutoff"]==c and z["n"]<=101]
        for ax,stat in zip(axes,("mean","median")):
            ax.plot([z["n"] for z in vals],[z[stat] for z in vals],style,color=color,ms=3,lw=1.1,label=label)
    for ax,title in zip(axes,("Arithmetic mean","Median")):
        ax.axvline(82,color=".55",lw=.8,ls="--")
        ax.set(yscale="log",xlabel="Training size",title=title)
        decorate(ax)
    axes[0].set_ylabel("Holdout MSE")
    axes[0].legend(fontsize=7,frameon=False,ncol=2)
    fig.savefig(output/"cutoffs.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=HERE)
    parser.add_argument("--output",type=Path,default=HERE/"summary")
    args=parser.parse_args()
    summarize(args.inputs,args.output)
