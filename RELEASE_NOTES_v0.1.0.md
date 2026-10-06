# v0.1.0 — Initial public reference implementation

These notes are prepared for the first release. A GitHub release and Zenodo archival are still pending at preparation time.

## Included

- GSMLP for flat-vector inputs and VIHON for samples composed of three-dimensional vectors.
- Corrected analytical gradients, documented preprocessing, explicit persistence and local random_state control.
- Pinned dependencies, executable examples, 70 regression tests, gradient audits and model manuals.
- BSD 3-Clause license and citation metadata for both software authors and foundational publications.

## Validation

Local checks on 2026-10-06 passed 31 GSMLP and 39 VIHON unittest methods, plus 252 and 1,752 analytical-gradient comparisons respectively. Tested environment: Python 3.12.8, NumPy 1.26.4, scikit-learn 1.6.1 and pandas 2.2.3. See validation/VALIDATION.md for configurations and limitations.

These results do not establish universal convergence, heuristic superiority or reproduction of historical benchmarks. The analytical-gradient audit is distinct from VIHON's optional heuristic-modified updates.

## Important conventions

- GSMLP training supports one hidden layer; VIHON requires two hidden layers.
- Epoch limits are inclusive; max_epoch=5 allows six epochs from a new instance.
- batch>1 means full-batch updates, not mini-batches of that size.
- The historical momentum and stopping conventions are retained and documented.
- VIHON preserves the fixed input-domain map [0,256] to [-1,1]; target scaling uses training targets only.
- Each instance supports one training run. Only load trusted pickle files.

## Credits

Juan Luis Crespo-Mariño: model creation, scientific responsibility and corresponding contact, Tecnológico de Costa Rica, jcrespo@itcr.ac.cr.
Aníbal Sancho-Theoduloz: development and debugging of the original Python implementations.

No software DOI is asserted in these notes. Publication DOIs in CITATION.cff identify the foundational papers, not this software release.
