# canalization-metrics-comparison
Diverse mathematical formalizations have been developed to capture
canalization in Boolean functions, based on the collective effects of inputs or their
redundancy. Since these definitions capture distinct but overlapping features of gene
regulatory networks, we sought to understand which metric is more representative of
network stability.

## Purpose
This repository contains all code to generate the figures presented in
> **Emma Tran, Ahana Ghosh, Ruodan Liu, Sean Campbell, Claus Kadelka.** *Comparing measures of collective canalization in Boolean functions and networks*. Under review.

## Repository structure
```
├── figures/                  
├── other/           
├── scripts/
├── tables/                    
├── requirements.txt
└── README.md
```

### figures/
Generated manuscript figures.

### scripts/
Source code used for all analyses.

## Installation
Clone the repository:
```bash
git clone https://github.com/KadelkaLab/canalization-metrics-comparison.git
cd canalization-metrics-comparison
```

Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing results
To reproduce all manuscript figures:
```bash
python can_strength_vs_redundancy_combined_metrics.py
python can_strength_vs_redundancy_function_metrics.py
python can_strength_vs_redundancy_network_metrics.py
```

Generated figures will be written to:
```
figures/
```

## Dependencies
- Python 3.11
- BoolForge
- Joblib
- Matplotlib
- NumPy
- pandas
- scikit-learn
- SciPy
- Seaborn

## Authors
- Emma Tran
- Ahana Ghosh
- Ruodan Liu
- Sean Campbell
- Claus Kadelka

## License
MIT License
