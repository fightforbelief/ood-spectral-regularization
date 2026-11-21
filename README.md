# When a single training subset sets the peak of a minimum-norm learning curve

[Manuscript (PDF)](paper.pdf) · [Manuscript source](manuscript/paper.tex) · [Paired losses and seeds](replication/losses.csv) · [Cutoff sensitivity](replication/summary/sensitivity.csv)

Yunjie Guo and Hanqing Yang, Independent Researcher, Suzhou, China.

At a training size of 73 on UCI Superconductivity, one of 100 subsets contributes 99.92% of the summed minimum-norm holdout loss. A post-hoc relative cutoff of 1e-12 changes just five of 1,500 fitted ranks. The arithmetic-mean peak shifts from 73 to 83, while the median peak stays at 81. A finite-field calculation certifies centered rank 71 for the extreme prefix in the recorded decimal features. The default solver retained 71 directions, and the alternative retained 70.

The original comparison used fixed holdouts for YearPredictionMSD and Superconductivity. At 15 sizes in each dataset, source-only cross-validated ridge has the smallest mean MSE of minimum norm, ridge and PCR. The training-mean reference, cutoff sweep, and algebraic checks were added after observing the original losses. All primary estimates include all 100 permutations. Pointwise paired intervals describe variability over those permutations, conditional on the holdout.

## Reproduce the paper from the released scores

Run from the repository root. The study used Python 3.12.10 on Windows. These commands use its recorded package versions. On another platform the equivalent interpreter is `.venv/bin/python`.

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py -m replication.summarize --output replication/summary
& $py -m replication.diagnostics
& $py -m pytest -q replication/test_rank.py
& $py manuscript/tables.py
tectonic --outdir . manuscript/paper.tex
```

The summary command checks complete paired grids for 3,000 training prefixes and 7,500 cutoff-specific scores. It writes risk curves, 120 paired interval rows (three estimator contrasts plus the training-mean comparison), the sensitivity table, and both figures. The diagnostic command joins the archived spectra to the losses, checks the exact-rank certificates against the spectral selection, and reports the effect of omitting each complete permutation. The PDF builds from the included tables and figures without rerunning any model.

## Rebuild fits from the UCI data

Download and extract the original [Superconductivity archive](https://archive.ics.uci.edu/dataset/464/superconductivty+data) and [YearPredictionMSD archive](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd). The commands below assume the two extracted tables are in `data/raw/`. In the public replication, the stored seed index is zero-based. Omitting both selectors from a replay command refits all 100 permutations at all 15 sizes, including the original source-only tuning.

```powershell
& $py -m replication.replay YearPredictionMSD data/raw/YearPredictionMSD.txt --repetition 7
& $py -m replication.replay Superconductivity data/raw/train.csv --repetition 5
& $py -m replication.cutoff data/raw/train.csv cutoff-recomputed.csv
```

The replay reconstructs the source order and inner folds, refits all three estimators, then checks MSE against the released records. The materials routines also verify the raw-file and ordered split hashes. The cutoff command checks every default loss before accepting its output. Its tolerance grid includes the default and floors of 1e-12, 1e-10, 1e-8 and 1e-6 times the largest singular value. The release reports float64 fits from SciPy GESDD. Near-null directions are sensitive to changes in numerical libraries. A replay discrepancy reports the affected fit for inspection.

## Verification on the study machine

Using the project environment on Windows, the original checkout passed 107 tests. A clean copy of the public analysis package passed all three exact-rank tests and recreated all three summary CSVs byte-for-byte. From the two raw UCI files, the public CV replay recovered all 90 estimator losses on the two full trajectories shown in the commands above, with maximum relative error zero. The separate cutoff reconstruction recovered all 1,500 default materials losses exactly. The archive exporter supplied the full 9,000 original model losses from the completed runs.

## Files and provenance

- [losses.csv](replication/losses.csv) contains one row per dataset, repetition and training size. The three estimator columns and the training-mean reference are holdout MSEs on the original response scale. Repetitions are paired within a dataset.
- [cutoff.csv](replication/cutoff.csv) contains the relative floor (zero means default), retained rank and holdout MSE. The last two columns are a modular lower bound and distinct-row upper bound on exact centered rank. They are populated for the 25 prefixes selected by the default spectral rule. The bounds repeat across cutoff values for a given prefix.
- The archived [music spectra](replication/spectra_YearPredictionMSD.csv) and [materials spectra](replication/spectra_Superconductivity.csv) record the near-boundary windows. [Provenance](replication/provenance.json) identifies the source manifests, protocols and materials split. [export_archives.py](replication/export_archives.py) reconstructs the compact loss file in the original two-checkout archive.
- The original YearPredictionMSD package is in [p19](p19), and the separate materials execution snapshot is in [superconductivity_arm](superconductivity_arm/README.md). Their original full test suite and formal run commands use historical local gate, protocol and audit files. The public replay uses the released data and scripts above. The [earlier manuscript](https://github.com/fightforbelief/ood-spectral-regularization/blob/7951bf5962c67251b6aa749c78457db415adf161/paper.pdf) contains the historical reporting and additional diagnostics.
