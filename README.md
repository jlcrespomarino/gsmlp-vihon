# GSMLP and VIHON

Reference Python implementations of neural-network architectures with trainable Gaussian synapses.

Status: initial public reference release v0.1.0, published on 2026-10-06 and archived in Zenodo. Passing the documented checks does not establish universal convergence or reproduce historical benchmarks.

- [Archived version v0.1.0](https://doi.org/10.5281/zenodo.23194204)
- [All versions (concept DOI)](https://doi.org/10.5281/zenodo.23194203)
- [GitHub release v0.1.0](https://github.com/jlcrespomarino/gsmlp-vihon/releases/tag/v0.1.0)

The archived release corresponds to commit 01a0aaad89eff074f34fc04ad9fed34fe7058c63. The main branch may contain later documentation or development changes. Use the archived version for an exact software snapshot.

## Authors and roles

- Juan Luis Crespo-Mariño: model creation, scientific responsibility and corresponding contact. Tecnológico de Costa Rica. [jcrespo@itcr.ac.cr](mailto:jcrespo@itcr.ac.cr).
- Aníbal Sancho-Theoduloz: development and debugging of the original Python implementations supplied for this distribution.

The foundational publications identify their historical author lists. Changes relative to the supplied Python files are documented in the model-specific histories.

## Models

| Model | Training input shape | Hidden layers | Input scaling |
|---|---|---|---|
| GSMLP | (n_samples, n_in) | One for training; more supported for forward evaluation only | MinMaxScaler fitted exclusively to training inputs |
| VIHON | (n_samples, n_in, 3) | Exactly two: vector-to-vector and vector-to-scalar | Fixed coordinate map x/128-1 for domain [0,256] |

Both fit target scaling exclusively to training targets. Each instance supports one training run. VIHON keeps analytical gradients separate from optional heuristic updates. Read the manuals for nonstandard momentum, full-batch and inclusive epoch conventions.

## Download and environment

Use GitHub's Code > Download ZIP and open the extracted folder, or clone:

```text
git clone https://github.com/jlcrespomarino/gsmlp-vihon.git
cd gsmlp-vihon
```

Python 3.12.8 was tested. Dependencies are pinned to the checked environment, not advertised as minimum supported versions. From the repository root, create a separate environment:

```text
python -m venv .venv
```

Activate it using the command for your shell:

| Shell | Command |
|---|---|
| Linux/macOS shell | source .venv/bin/activate |
| Windows PowerShell | .\.venv\Scripts\Activate.ps1 |
| Windows Command Prompt | .venv\Scripts\activate.bat |

Install the core dependencies and run the examples:

```text
python -m pip install -r requirements.txt
python examples/gsmlp_minimal_example.py
python examples/vihon_minimal_example.py
```

This is a source-module distribution, not an installable PyPI package. The example scripts configure their own source import path. For your own script run from the repository root:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('src').resolve()))
from gsmlp import GSMLP
from vihon import VIHON
```

Targets must be two-dimensional, including single-output tasks. Predictions accept one sample at a time. Model saving is explicit; only load trusted pickle files.

## Run all checks

```text
python -m pip install -r requirements-validation.txt
python run_checks.py
```

The validation requirements include the core requirements and pandas. pandas is not needed by the models themselves. The runner checks that both suites are present, imports both models, runs 70 unittest methods, and writes separate gradient-audit CSVs under validation: 252 GSMLP comparisons and 1,752 VIHON comparisons in the documented environment. A failed test or audit exits unsuccessfully.

[Validation results and scope](validation/VALIDATION.md) explains configurations, tolerances and limitations. The gradient checks concern analytical derivatives, not the efficacy or equivalence of heuristic-modified optimizer updates.

## Documentation

- [GSMLP manual](docs/gsmlp.md)
- [VIHON manual](docs/vihon.md)
- [GSMLP history](docs/CHANGELOG_GSMLP.md)
- [VIHON history](docs/CHANGELOG_VIHON.md)
- [Joint changelog](CHANGELOG.md)
- [Scientific references](docs/references.md)
- [Contribution guidelines](CONTRIBUTING.md)

## Citation and license

Please cite the archived software version and the relevant foundational publication when using these implementations in research. [CITATION.cff](CITATION.cff) provides software citation metadata.

Crespo-Mariño, J. L., & Sancho-Theoduloz, A. (2026). *GSMLP and VIHON: Gaussian-Synapse Neural Networks in Python* (Version v0.1.0) [Computer software]. Zenodo. [https://doi.org/10.5281/zenodo.23194204](https://doi.org/10.5281/zenodo.23194204).

Use DOI 10.5281/zenodo.23194204 to cite the exact v0.1.0 snapshot. DOI 10.5281/zenodo.23194203 represents all versions and resolves to the latest one.

GSMLP: Duro, R. J., Crespo, J. L., and Santos, J. (1999). Training Higher Order Gaussian Synapses. LNCS 1606, 537–545. [DOI: 10.1007/BFb0098211](https://doi.org/10.1007/BFb0098211).

VIHON: Crespo, J. L., and Duro, R. J. (2005). Considering Multidimensional Information Through Vector Neural Networks. LNCS 3512, 17–24. [DOI: 10.1007/11494669_3](https://doi.org/10.1007/11494669_3).

The DOIs listed with the two foundational papers identify publications, not software releases. BSD 3-Clause; see [LICENSE](LICENSE).
