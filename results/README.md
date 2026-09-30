# Aggregate experimental results

- `aggregate_metrics.csv`: 18 historical configurations, one grouped random split.
- `time_validation_metrics.csv`: later-date holdout, after selection on development data.
- `search_results.csv` and `cv_fold_metrics.csv`: 39 candidates and 156 fold fits.
- `cv_splits.csv`: chronological fold boundaries and row counts, no individual records.
- `environment_sensitivity.csv`: illustrative perturbations, not measured forecast error.
- `browser_verification.json`: browser parity and interface checks for this export.

The two evaluation schemes use different test sets; their scores are not directly
comparable. These result CSVs omit row-level observations and predictions. The repository
currently tracks the original Excel workbook at its root, so publishing this
repository would also publish those records. Units and limitations are described
in the root README and the website's About page.
