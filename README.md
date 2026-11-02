# Regularization near the interpolation threshold

Empirical comparison of minimum-Euclidean-norm least squares, source-only cross-validated ridge regression, and source-only cross-validated principal components regression. The paper is [`paper.pdf`](paper.pdf).

Two fixed holdouts are used. YearPredictionMSD has 90 predictors and an artist-disjoint holdout. Superconductivity has 81 predictors and an exact-feature-grouped holdout. Each benchmark uses 100 training-subset permutations and 15 sample sizes. Ridge has the lowest mean holdout mean squared error at every archived size on both benchmarks, and all 30 pointwise paired intervals for PCR minus ridge exclude zero. Because features are centered on the training data, the interpolation boundary lies at n = p + 1. On Superconductivity, one repetition contributes 99.92% of the total minimum-norm loss at n = 73, so the mean and median curves peak at different sizes. On YearPredictionMSD, ridge improves on a training-mean predictor by only about 3% near the threshold. Across these two holdouts, regularization mitigates severe near-threshold risk. The experiments do not establish a universal ranking, and they do not identify a causal effect of distribution shift.

## Run

From this directory, with the project virtual environment:

```text
python -m pytest -q
python -m p19.cli environment-audit
python -m p19.cli run-l2 --confirmatory
```

`run-l2 --confirmatory` refuses to run unless the local YearPredictionMSD approval files are present. The Superconductivity confirmatory package is the separate snapshot in [`superconductivity_arm/`](superconductivity_arm/README.md). Raw UCI tables and confirmatory prediction artifacts are not in this repository.
