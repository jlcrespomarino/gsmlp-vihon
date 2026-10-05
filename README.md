# GSMLP and VIHON

Reference Python implementations of **GSMLP** and **VIHON**, two high-order
neural-network architectures based on trainable Gaussian synapses.

> **Status:** Initial public reference implementation under preparation.
> The repository will include curated Python 3 source code, executable examples,
> tests, and documentation for reproducible use.

## Overview

This repository provides implementations of two related neural-network models:

- **GSMLP** (Gaussian Synapse Multilayer Perceptron) is designed for flat,
  vector-valued inputs. Its synapses are parameterized by trainable Gaussian
  functions.

- **VIHON** is designed for inputs whose individual elements are
  three-dimensional vectors. It processes multidimensional vector information
  directly through vector-valued intermediate representations.

Both models use Gaussian-type synaptic functions with trainable parameters.
The public release will document the architectures, training procedures,
input conventions, software requirements, examples, and tests.

## Authors and roles

- **Juan Luis Crespo-Mariño**  
  Creator of the GSMLP and VIHON neural-network models; scientific design;
  algorithmic foundations; corresponding author.  
  Tecnológico de Costa Rica  
  Contact: [jcrespo@itcr.ac.cr](mailto:jcrespo@itcr.ac.cr)

- **Aníbal Sancho-Theoduloz**  
  Development, debugging, and preparation of the Python implementations
  distributed in this repository.

The GSMLP and VIHON models were originally developed by Juan Luis
Crespo-Mariño. The Python reference implementations distributed here were
developed and debugged by Aníbal Sancho-Theoduloz under the scientific
direction of Juan Luis Crespo-Mariño.

## Scientific background

### GSMLP

GSMLP is based on high-order Gaussian synapses. A representative synaptic
weight function is parameterized as:

\[
w(x) = A \exp\left(B(x - C)^2\right),
\]

where \(A\), \(B\), and \(C\) are trainable parameters. The training procedure
extends backpropagation to update these synaptic parameters.

The foundational publication is:

> Duro, R. J., Crespo, J. L., and Santos, J. (1999). *Training Higher Order
> Gaussian Synapses*. In *Foundations and Tools for Neural Modeling*,
> Lecture Notes in Computer Science, vol. 1606, pp. 537–545.
> Springer. https://doi.org/10.1007/BFb0098211

### VIHON

VIHON extends the treatment of high-order Gaussian synapses to inputs
structured as three-dimensional vectors. It provides a vector-neural
architecture and an associated backpropagation adaptation for directly
processing multidimensional information.

The foundational publication is:

> Crespo, J. L., and Duro, R. J. (2005). *Considering Multidimensional
> Information Through Vector Neural Networks*. In *Computational Intelligence
> and Bioinspired Systems*, Lecture Notes in Computer Science, vol. 3512,
> pp. 17–24. Springer. https://doi.org/10.1007/11494669_3

## Repository structure

The repository is being organized as follows:

```text
src/
  gsmlp.py                    # GSMLP implementation
  vihon.py                    # VIHON implementation
examples/
  gsmlp_minimal_example.py    # GSMLP demonstration
  vihon_minimal_example.py    # VIHON demonstration
tests/
  test_gsmlp.py               # GSMLP checks
  test_vihon.py               # VIHON checks
docs/
  gsmlp.md                    # GSMLP documentation
  vihon.md                    # VIHON documentation
  references.md               # Bibliographic references
```

## Installation

The installation instructions and dependency versions will be included with
the first curated public release.

## Citation

Citation metadata will be available in `CITATION.cff`. If you use GSMLP or
VIHON in academic work, please cite both this software repository and the
relevant foundational publication listed above.

## License

This project is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for details.
