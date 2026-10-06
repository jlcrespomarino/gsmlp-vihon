# Changes from supplied VIHON-2.py

## Python 3 release candidate (unreleased)

- Python 3 syntax/type adaptation and removal of unused plotting imports.
- Correct output delta sign, 1/n_out loss normalization, downstream chain-rule factor and vB coordinate indices.
- Analytical gradients separated from optional center-surrogate/width heuristics; legacy_heuristics defaults to True and branch counters expose decisions.
- Authorized second-layer condition fix: exp(B*dot(h-C,h-C)) in both learning-rate branches.
- Fixed input domain [0,256] retained; target scaler fitted only on training samples after a fixed split.
- Local random_state and stored split indices; legacy per-epoch resplitting rejected.
- Stable sigmoid, numerical guards, public argument validation and zero-loss denominator handling.
- One fit per instance, explicit failed-run policy and successful-training requirement for prediction/checkpoint saving.
- Automatic periodic AND final saves removed; explicit save_model/load_model.
- Literal-only vector CSV parsing; optional explicit prediction CSV output; portable long-form numeric weight CSV.
- Historical nonstandard momentum, full-batch interpretation of batch>1, last-sample heuristic decisions, inclusive epoch limit, four-change stopping and first/second-layer-only positive-B resets retained and documented.

Architecture and Gaussian-synapse forward equations are preserved. Corrected training need not reproduce numerical trajectories of the supplied flawed implementation. No historical benchmark replication is claimed.
