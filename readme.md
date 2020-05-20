## Automated coding using NTEE and supervised machine-learning: A guide and benchmark

We achieved **90%** overall accuracy for classifying the nonprofits into 9 broad categories, and **88%** for classifying them into 25 major groups. The intercoder reliabilities between algorithms and human coders measured by kappa statistics are in the "almost perfect" range of 0.80–1.00. Take a look at [a sample result](/#).

### Useful resources
- [API](/API/) for classifying text descriptions of nonprofits into [NTEE codes](https://nccs.urban.org/project/national-taxonomy-exempt-entities-ntee-codes#overview).
- [Universal Classification Files](/dataset/UCF) for benchmarking and testing.
- [Methodology paper](/#).

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
