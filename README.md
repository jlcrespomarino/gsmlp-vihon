# GSMLP and VIHON

Reference Python implementations of GSMLP and VIHON, neural-network architectures with trainable Gaussian synapses.

## Publication status

Publication is in progress. The jointly checked source modules, tests, examples and manuals are awaiting incorporation in subsequent commits. No tagged software release or software DOI is declared yet.

Local validation of the prepared joint snapshot passed 70 unittest methods and 2,004 analytical-gradient comparisons: 252 for GSMLP and 1,752 for VIHON. These results do not establish universal convergence or reproduce all historical experiments. Installation instructions will accompany the code publication.

## Authors and roles

- Juan Luis Crespo-Mariño: model creation, scientific responsibility and corresponding contact. Tecnológico de Costa Rica. [jcrespo@itcr.ac.cr](mailto:jcrespo@itcr.ac.cr).
- Aníbal Sancho-Theoduloz: development and debugging of the original Python implementations supplied for this distribution.

The foundational publications below identify their historical author lists. The accompanying model-specific change histories will document mathematical corrections and software-release changes relative to the supplied Python files.

## Models

GSMLP accepts flat-vector samples. Training supports one hidden layer; additional hidden layers support forward evaluation only. Input scaling is fitted exclusively to training inputs.

VIHON accepts samples composed of three-dimensional vectors and uses two hidden layers. Its fixed coordinate input map preserves the domain [0,256]. Target scaling is fitted exclusively to training targets in both implementations. VIHON separates analytical gradients from optional documented heuristic update policies.

## Scientific references

Duro, R. J., Crespo, J. L., and Santos, J. (1999). Training Higher Order Gaussian Synapses. Foundations and Tools for Neural Modeling, Lecture Notes in Computer Science 1606, 537–545. [DOI: 10.1007/BFb0098211](https://doi.org/10.1007/BFb0098211).

Crespo, J. L., and Duro, R. J. (2005). Considering Multidimensional Information Through Vector Neural Networks. Computational Intelligence and Bioinspired Systems, Lecture Notes in Computer Science 3512, 17–24. [DOI: 10.1007/11494669_3](https://doi.org/10.1007/11494669_3).

## Citation and license

Please cite this software repository and the relevant foundational publication when using these implementations in research. [CITATION.cff](CITATION.cff) supplies software citation metadata. Until a tagged release and DOI exist, record the commit used to identify the software snapshot. Publication DOIs are not software DOIs.

BSD 3-Clause. See [LICENSE](LICENSE).
