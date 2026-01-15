# Tennis Match Prediction with Machine Learning

## Overview

This project develops machine learning models to predict professional ATP tennis match outcomes and evaluates their profitability when applied to betting strategies. We use 21 years of ATP historical data (2003–2024) and engineer 123 features capturing player skill dynamics, then compare three model classes: logistic regression, XGBoost, and multilayer perceptrons.

## Research Question

Can machine learning models predict tennis match outcomes with sufficient accuracy to generate positive returns when applied to betting strategies against bookmaker odds?

## Dataset

- **Source:** [Jeff Sackmann's ATP Tennis Dataset](https://github.com/JeffSackmann/tennis_atp)
- **Betting Odds:** [Tennis-Data.co.uk](http://www.tennis-data.co.uk/)
- **Period:** 2003–2024 (58,833 ATP matches)
- **Train/Test Split:** Temporal split at 2024 US Open

## Features (123 total)

| Category | Description |
|----------|-------------|
| **Static attributes** | ATP ranking, ranking points, height, age differences |
| **Head-to-head** | Historical win/loss records overall and per-surface |
| **Form** | Win rates over rolling windows (3, 5, 10, 25, 50, 100 matches) |
| **Elo ratings** | Overall and surface-specific ratings with adaptive K-factors and inactivity adjustments |
| **Rolling statistics** | Ace rate, break points saved, return points won, etc. across multiple time windows |

## Models

1. **Logistic Regression** — L2-regularised baseline
2. **XGBoost** — Two variants tuned for log-loss and error rate
3. **Multilayer Perceptron** — Two variants tuned for log-loss and error rate

Hyperparameter optimisation performed using Optuna with TPE sampler and TimeSeriesSplit cross-validation.

## Betting Strategies

Evaluated on 126 matches from the 2024 US Open using Pinnacle Sports closing odds:

1. **Winner Prediction Strategy:** Fixed stake on the predicted winner
2. **Threshold-Hedged Strategy:** Proportional stakes based on model probabilities, betting only when the model-implied edge exceeds the bookmaker margin

## Results

| Model | Test Accuracy | Strategy 1 ROI | Strategy 2 ROI |
|-------|---------------|----------------|----------------|
| Logistic Regression | 65.3% | -5.07% | -2.88% |
| XGBoost (Log-Loss) | 67.2% | +0.48% | -4.55% |
| XGBoost (Error) | 66.9% | -2.67% | -4.08% |
| MLP (Log-Loss) | 66.4% | +1.06% | -5.27% |
| MLP (Error) | 66.1% | +0.88% | -4.80% |

## Repository Structure
```
├── CODE/
│   ├── DATA/                   # Raw ATP match CSVs from Sackmann (1991–2024)
│   ├── Final/
│   │   ├── betting_order.csv   # Alignment file for test set to betting matches
│   │   ├── order.csv           # Match ordering reference
│   │   ├── test.csv            # Test set: 705 matches from 2024 US Open onwards
│   │   └── usopen.csv          # 2024 US Open betting odds from Tennis-Data.co.uk
│   ├── Data.ipynb              # EDA, feature engineering, train/test set creation
│   ├── Log_Model.ipynb         # Logistic regression baseline (no hyperparameter tuning)
│   ├── XGBoost.ipynb           # XGBoost with Optuna hyperparameter optimisation and CV
│   ├── MLP.ipynb               # MLP with Optuna hyperparameter optimisation and CV
│   └── Betting.ipynb           # Final evaluation: train on full set, test on 705, betting strategies on 126 US Open matches, 10 metrics across 5 models × 2 strategies
├── .gitignore
└── README.md
```

**Note:** `train.csv` is not included due to file size constraints. Run `Data.ipynb` to generate it.

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*
- Wilkens, S. (2021). Sports prediction and betting models in the machine learning age: The case of tennis. *Journal of Sports Analytics*
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML 2005*. 625-632.
- Wilson, A., Roelofs, R., Stern, M., Srebro, N. & Recht, B. (2017). The Marginal Value of Adaptive Gradient Methods in Machine Learning. *arXiv:1705.08292*
- Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR* 15, 1929-1958.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. (2017). On Calibration of Modern Neural Networks. *ICML 2017*
- Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *arXiv:2106.11959*
- Shmuel, A., Glickman, O. & Lazebnik, T. (2025). A comprehensive benchmark of machine and deep learning models on structured data. *Neurocomputing* 655, 131337.
- Kingma, D. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *ICLR 2015*

## Acknowledgements

We thank Jeff Sackmann for providing the tennis match dataset and Tennis-Data.co.uk for historical betting odds data.

## License

MIT License
