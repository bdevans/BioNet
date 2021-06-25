# BioNet
Deep Convolutional Neural Networks with bio-inspired filters. 

1. Clone this repository
2. Clone the CIFAR-10G generalisation test set
3. Optionally clone an ALL-CNN implementation
4. Set your `project_dir` in the notebook and pass your `data_dir` (`ln -s /shared/data/ data`)

Expected directory structure
----------------------------

.
├── blah.py
├── bionet
│   ├── config.py
│   ├── explain.py
│   ├── __init__.py
│   ├── plots.py
│   └── preparation.py
├── data
│   ├── CIFAR-10G
│   ├── ecoset
│   └── ecoset-cifar10
├── logs
├── models
├── notebooks
├── results
├── scripts
├── model.py
└── README.md