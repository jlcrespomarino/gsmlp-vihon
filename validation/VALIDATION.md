# Joint validation results

## Environment and execution

The checks were executed locally in a fresh Python process on 2026-10-06 using Python 3.12.8, NumPy 1.26.4, scikit-learn 1.6.1 and pandas 2.2.3. No claim of a GitHub Actions run is made.

The checker and audit file identities match the published versions. The reviewed source modules and suites have also been checked against their published Git blob identifiers. Only the suites' model-specific audit imports differ from the original individual-package files.

[Execution summary](combined_results.txt) records the checker stdout. [Machine-readable summary](summary.json) records counts and discrepancies. This publication includes the summary, not a complete unittest stderr transcript.

## Results

| Check | Result |
|---|---|
| GSMLP unittest methods | 31 passed |
| VIHON unittest methods | 39 passed |
| Combined unittest methods | 70 passed |
| GSMLP gradient comparisons | 252 passed; maximum absolute discrepancy 6.5059916479269335e-12 |
| VIHON gradient comparisons | 1,752 passed; maximum absolute discrepancy 3.084937839036017e-11 |

Parameterized subchecks are not counted as separate unittest methods. Gradient comparisons are repeated parameter/configuration checks, not 2,004 independent network parameters.

## Gradient audit configurations

GSMLP: architectures 2–2–1 and 2–2–2, seeds 0, 7 and 42, central-difference steps 1e-5 and 1e-6. The audit extracts the gradient statements used by train().

VIHON: architectures 2–2–2–1 and 2–2–2–2, the same seeds and steps, both legacy_heuristics settings. The audit calls the actual analytical accumulation method used by training. Each setting contributes 876 comparisons.

Tolerance: rtol=1e-4 and atol=1e-7. Correct analytical gradients are distinct from heuristic-modified updates. Specific suite tests exercise active/inactive heuristic branches, corrected distance conditions, positive-B resets and momentum behavior.

## Coverage and limitations

The suites cover argument and data validation, finite outputs, numerical extremes, lifecycle rules, reproducibility, holdout isolation, schedules, batch and stopping conventions, persistence and CSV exports. They use small deterministic synthetic datasets in one tested environment.

Passing does not prove exhaustive correctness, convergence on arbitrary datasets, heuristic superiority, cross-platform identity or historical benchmark reproduction. Failed numerical training can leave partial state; discard the instance. Only load trusted pickle files. See the model manuals for retained nonstandard optimizer conventions.

## Reproduce

From the repository root, install requirements-validation.txt and run python run_checks.py. The runner regenerates gradient_check_gsmlp.csv and gradient_check_vihon.csv under validation. Raw CSVs are reproducible outputs and are not included in this documentation commit.
