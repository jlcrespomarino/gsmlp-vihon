# Changes from supplied GSMLP.py

## Python 3 release candidate (unreleased)

- Python 3 syntax and numerical type checks; no import-time Boston dataset load or plotting dependencies.
- Output-gradient multiplication replaces exponentiation; corrected output delta in hidden backprop; consistent 1/n_out loss normalization.
- Fixed holdout split before fitting both scalers; legacy per-epoch resplitting explicitly rejected.
- Local random_state for initialization and split, and stored split indices.
- Stable sigmoid and guarded numerical arithmetic, explicit argument checks and a zero-loss denominator guard.
- Explicit save_model/load_model replaces automatic periodic AND final checkpoints.
- Explicit CSV output path; portable weight-export paths and documented long-form weight CSV format.
- Repeated training is rejected to avoid re-fitting scalers while retaining old weights; failed training instances must be discarded.
- Forward evaluation returns a copy; supports multiple hidden layers. Training remains limited to one hidden layer.
- Original momentum formula, summed full-batch gradients, full-batch interpretation of batch>1, inclusive epoch limit and four-change stopping rule retained and documented.

These changes affect execution and numerical training trajectories. This is a corrected implementation, not a claim of bitwise equivalence to the supplied code or reproduction of all historical results. Architecture and Gaussian synapse equations are unchanged.
