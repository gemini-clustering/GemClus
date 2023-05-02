# GEMCLUS - A package for discriminative clustering using GEMINI

The **gemclus**  package provides simple tools to perform discriminative clustering using the generalised mutual
information (GEMINI).
The package was written to be a scikit-learn compatible extension.

You can find the complete documentation of the package here: `Link to be announced`

## Installation

Use the following instruction for installing the package:

```commandline
pip install gemclus
```

The library requires a couple scientific package to run:

+ NumPy
+ Scipy
+ POT
+ Scikit-learn

## Reference

If this work helped you, please cite our original NeurIPS work:

```
Ohl, L., Mattei, P. A., Bouveyron, C., Harchaoui, W., Leclercq, M., Droit, A., & Precioso, F.
(2022, October).
Generalised Mutual Information for Discriminative Clustering.
In Advances in Neural Information Processing Systems.
```

or

```bibtex
@inproceedings{ohl2022generalised,
title={Generalised Mutual Information for Discriminative Clustering},
author={Louis Ohl and Pierre-Alexandre Mattei and Charles Bouveyron and Warith Harchaoui and Micka{\"e}l Leclercq and Arnaud Droit and Frederic Precioso},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=0Oy3PiA-aDp}
}
```

## Acknowledgements

This work has been supported by the French government, through the 3IA Côte d'Azur, Investment in the Future, project
managed by the National Research Agency (ANR) with the reference number ANR-19-P3IA-0002. We would also like to thank
the France Canada Research Fund (FFCR) for their contribution to the project. This work was partly supported by
EU Horizon 2020 project AI4Media, under contract no. 951911.

Also many many thanks to Pierre-Alexandre Mattei, Frederic Precioso and Charles Bouveyron for their contribution
in the GEMINI project. Thanks as well go to Jhonatan Torres for his insights on the development.
