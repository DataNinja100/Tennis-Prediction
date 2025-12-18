# Tennis Match Prediction with Neural Networks

## Project Goal

This project develops a neural network model to predict professional tennis match outcomes using 34 years of ATP historical data (1991–2025). We engineer 79 features capturing player skill dynamics and evaluate two distinct betting strategies against Grand Slam bookmaker odds: (1) backing the model's predicted winner outright, versus (2) hedging positions based on discrepancies between model-implied probabilities and market odds. The core research question is whether our model can identify market inefficiencies and achieve positive expected value.

---

## Report Structure

| Section | Content |
|---------|---------|
| **Introduction** | Justify importance and challenges of the topic; review related work, identify shortcomings, and motivate proposed solution; summarize contributions and novelty |
| **Problem Formulation** | Self-contained notations and definitions; mathematical formulation of the prediction and betting problem |
| **Proposed Solution** | Describe neural network architecture and methodology; discuss computational complexity |
| **Numerical Experiments** | Present benchmarks and preprocessing; detail performance measures; design and discuss experiments; compare with alternative methods |
| **Conclusions** | Point to limitations; motivate future research directions |

I will add a google doc for us
---

## Code Structure

### 1. Data Cleaning, Feature Engineering & EDA (DONE)

- Clean raw ATP match data (1991–2025)
- Engineer 79 features:
  - Surface-specific ELO ratings (Hard, Clay, Grass, Carpet) with experience-adjusted K-factors
  - Multi-scale rolling statistics (windows: 3–2,000 matches)
  - Head-to-head records, serving stats, break point conversion
  - Form indicators, fatigue measures, "rust" adjustments
- Exploratory data analysis
- Output: final training dataset

### 2. Model Development

- Build neural network architecture on prepared dataset
- Hyperparameter optimization via grid search on validation set
- Temporal train-validation-test split (no look-ahead bias)
- Output: trained model with optimal hyperparameters

### 3. Betting Strategy Evaluation

- Obtain betting odds from [tennis-data.co.uk](http://www.tennis-data.co.uk/)
- Aggregate bookmaker odds or select most popular bookmaker
- Clean and filter to latest Grand Slam tournament
- Convert odds to implied probabilities
- Test two strategies:
  1. **All-in on predicted winner:** Bet on model's highest-probability outcome
  2. **Hedge based on model probabilities:** Size positions according to edge between model and market probabilities
- Evaluate: accuracy vs. odds baseline, ROI, expected value analysis
