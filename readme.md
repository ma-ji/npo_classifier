## Automated coding using machine-learning and remapping the U.S. nonprofit sector: A guide and benchmark

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

Ma, Ji. 2020. "Automated Coding Using Machine-Learning and Remapping the U.S. Nonprofit Sector: A Guide and Benchmark." _Nonprofit and Voluntary Sector Quarterly_ forthcoming.

```
@article{MaAutomatedcodingusing2020,
  title = {Automated Coding Using Machine-Learning and Remapping the {{U}}.{{S}}. Nonprofit Sector: {{A}} Guide and Benchmark},
  author = {Ma, Ji},
  date = {2020},
  journaltitle = {Nonprofit and Voluntary Sector Quarterly},
  volume = {forthcoming},
  url = {https://github.com/ma-ji/npo_classifier},
}
```

### Funding

This project was supported in part by the 2019-20 PRI Award and Stephen H. Spurr Centennial Fellowship from the [LBJ School of
Public Affairs](https://lbj.utexas.edu/) and a [Planet Texas 2050](https://bridgingbarriers.utexas.edu/planet-texas-2050/) grant from UT Austin.