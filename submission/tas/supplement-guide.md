# Replicating the learning curves

Start in the extracted folder containing this guide. The online appendix lists the full risk curves. All scores are mean squared errors on the fixed holdout, in squared years or squared kelvin. Each of the 100 repetitions has a stored seed and a nested sequence of 15 training sizes. Repetition indices start at zero.

## From the included data

The study environment used Python 3.12.10 on Windows. Create an environment with Python 3.12, install the versions listed in requirements.txt and run these commands using that interpreter. The same module entry points work on other operating systems.

    python -m pip install -r requirements.txt
    python -m replication.summarize --output reproduced
    python -m replication.diagnostics
    python -m pytest -q replication/test_rank.py

Compare reproduced/risk.csv, paired.csv and sensitivity.csv with their counterparts in replication/summary/. The two generated PDFs correspond to Figures 1 and 2. The sensitivity table supplies the peak locations and largest shares in Table 1. Running python manuscript/tables.py typesets Table 1 and the appendix rows from the included summary CSVs. The diagnostics command reports the spectral checks, exact-rank comparisons and results of leaving each training trajectory out.

- losses.csv contains 3,000 rows pairing minimum-norm, ridge, principal components and training-mean losses by dataset, repetition, and training size.
- cutoff.csv contains retained rank and MSE at each of five relative floors. Zero selects the solver default. The last two columns are the finite-field lower bound and the distinct-row upper bound for the 25 selected training prefixes.
- spectra_YearPredictionMSD.csv and spectra_Superconductivity.csv contain the archived near-boundary diagnostics. provenance.json identifies the original runs, inputs and ordered materials split by checksum.

These files are in replication/. The pinned dependencies apply to recomputation. Reading the CSVs and the appendix requires no software installation.

## From the raw observations

Obtain the original [music data](https://doi.org/10.24432/C50K61) and [materials data](https://doi.org/10.24432/C53P47) from UCI and extract the two specified tables. Provide their paths to the replay commands. The example trajectories include the influential fits. Omitting the repetition selector evaluates the whole benchmark.

    python -m replication.replay YearPredictionMSD path/to/YearPredictionMSD.txt --repetition 7
    python -m replication.replay Superconductivity path/to/train.csv --repetition 5
    python -m replication.cutoff path/to/train.csv cutoff-recomputed.csv

The materials commands verify the raw-file and split hashes. The cutoff replay checks the default loss in each prefix before releasing the completed output. The original SciPy GESDD calculations used float64. Changing numerical libraries can affect near-singular fits. The archived score files provide an exact reference for comparison. No API credentials or private research files are needed.
