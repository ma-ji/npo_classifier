## Automated coding using machine-learning and remapping the U.S. nonprofit sector: A guide and benchmark

[![NVSQ DOI](https://img.shields.io/badge/NVSQ%20DOI-10.1177/0899764020968153-brightgreen)](https://doi.org/10.1177/0899764020968153)
[![OPEN ACCESS PAPER](https://img.shields.io/badge/OPEN%20ACCESS%20PAPER@OSF-10.31219%2FOSF.IO%2FPT3Q9-blue)](https://dx.doi.org/10.31219/osf.io/pt3q9)

This research developed a machine-learning classifier that reliably automates the coding process using the National Taxonomy of Exempt Entities as a schema and remapped the U.S. nonprofit sector. I achieved 90% overall accuracy for classifying the nonprofits into nine broad categories and 88% for classifying them into 25 major groups. The intercoder reliabilities between algorithms and human coders measured by kappa statistics are in the "almost perfect" range of 0.80--1.00. The results suggest that a state-of-the-art machine-learning algorithm can approximate human coders and substantially improve researchers' productivity. I also reassigned multiple category codes to over 439 thousand nonprofits and discovered a considerable amount of organizational activities that were previously ignored. The classifier is an essential methodological prerequisite for large-N and Big Data analyses, and the remapped U.S. nonprofit sector can serve as an important instrument for asking or reexamining fundamental questions of nonprofit studies.

### Useful resources
- [Methodology paper](https://osf.io/pt3q9/).
- [API](/API/) for classifying text descriptions of nonprofits using [NTEE codes](https://nccs.urban.org/project/national-taxonomy-exempt-entities-ntee-codes#overview).
- [Universal Classification Files](/dataset/UCF) for benchmarking and testing.
- [Remapped U.S. nonprofit sector (i.e., nonprofits multi-labeled)](https://jima.me/?ntee_remap).
- Nonprofit Classifier Competition (TBD)

### Folder structure
```
.
├── API
├── dataset
│   ├── UCF
│   │   ├── test
│   │   └── train
│   ├── intermediary
│   └── muy060_suppl_supplementary_appendix
├── output
│   ├── classification_results
│   ├── fig
│   └── result_dicts
├── reference
│   ├── algorithms
│   └── assign_NTEE
└── script
    ├── classification_algorithms
    ├── data_acquisition
    └── data_analysis
```

### How to cite

Ma, J. (2021). Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector: A Guide and Benchmark. _Nonprofit and Voluntary Sector Quarterly, 50_(3), 662–687. https://doi.org/10.1177/0899764020968153

```
@article{MaAutomatedCodingUsing2021,
	title = {Automated {Coding} {Using} {Machine} {Learning} and {Remapping} the {U}.{S}. {Nonprofit} {Sector}: {A} {Guide} and {Benchmark}},
	volume = {50},
	issn = {0899-7640},
	shorttitle = {Automated {Coding} {Using} {Machine} {Learning} and {Remapping} the {U}.{S}. {Nonprofit} {Sector}},
	url = {https://doi.org/10.1177/0899764020968153},
	doi = {10.1177/0899764020968153},
	abstract = {This research developed a machine learning classifier that reliably automates the coding process using the National Taxonomy of Exempt Entities as a schema and remapped the U.S. nonprofit sector. I achieved 90\% overall accuracy for classifying the nonprofits into nine broad categories and 88\% for classifying them into 25 major groups. The intercoder reliabilities between algorithms and human coders measured by kappa statistics are in the “almost perfect” range of .80 to 1.00. The results suggest that a state-of-the-art machine learning algorithm can approximate human coders and substantially improve researchers’ productivity. I also reassigned multiple category codes to more than 439,000 nonprofits and discovered a considerable amount of organizational activities that were previously ignored. The classifier is an essential methodological prerequisite for large-N and Big Data analyses, and the remapped U.S. nonprofit sector can serve as an important instrument for asking or reexamining fundamental questions of nonprofit studies. The working directory with all data sets, source codes, and historical versions are available on GitHub (https://github.com/ma-ji/npo\_classifier).},
	language = {en},
	number = {3},
	urldate = {2021-05-22},
	journal = {Nonprofit and Voluntary Sector Quarterly},
	author = {Ma, Ji},
	month = jun,
	year = {2021},
	note = {Publisher: SAGE Publications Inc},
	keywords = {BERT, computational social science, machine learning, National Taxonomy of Exempt Entities, neural network, nonprofit organization},
	pages = {662--687}
}
```

### Funding

This project was supported in part by the 2019-20 PRI Award and Stephen H. Spurr Centennial Fellowship from the [LBJ School of
Public Affairs](https://lbj.utexas.edu/) and a [Planet Texas 2050](https://bridgingbarriers.utexas.edu/planet-texas-2050/) grant from UT Austin.