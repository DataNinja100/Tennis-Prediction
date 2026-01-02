# Tennis Match Prediction with Neural Networks


## Topic
- Accuracy vs Calibration: Evaluating Neural Network Predictions for Decision-Making in Tennis Betting Markets
## Research Question
- How does optimizing neural networks for predictive accuracy versus probability calibration affect decision quality and profitability in sports betting markets?


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

**Logistic regression baseline**
- Standard logistic regression model benchmark to compare against neural network

**XGBoost model**
- Comparison for Neural Network performance benchmark
- XGBoost is known for strong tabular data performance in different studies we look at probability distribution calibration which is important for decision making in betting
- one model optimized for log-loss and one for error-rate
- we look at expected calibration error (ECE) as an important metric for calibration


**Neural Network Model (MLP)**

- Feedforward neural network (MLP) with PyTorch
- 123(input) - 256 - 256 - 256 - 1(output) architecture with ReLU activations (THESE CAN BE EDITED IF ANY SUGGESTIONS) note that we do niot use sigmoid for final layer as we use BCEWithLogitsLoss which combines sigmoid and binary cross-entropy loss in a numerically stable way (Aslo returns logits neccesary for scaling later)
- Two models optimized for log-loss and error-rate respectively comparing their error rate log-loss and calibration performance(ECE)
- optimization with Adam optimizer, and optuna: Adam optimizes model weights; Optuna optimizes hyperparameters
- Optuna simulation 50 times (3-fold timeseries cross validation) for hyperparameter tuning (learning rate, weight decay, batch size, dropout rate, epochs) RANGES CAN BE EDITTED IF ANY SUGGESTIONS
- evalute loggloss optimized model and error-rate optimized model on 5 fold validation (CAN BE INCREASED ... my computer is too slow)
- for each hyperparemeter we plot their effect on test/ out of fold log-loss and error-rate and ECE +max normalization (but quite irrelevant results wise)
- FInally calibration of both models with temperature scaling and platt scaling to see imporvements in calibration (ECE) and log-loss
- Final evaluation metrics: log-loss, accuracy, error-rate, ECE



### 3. Betting Strategy Evaluation (TODO)

- Obtain betting odds from [tennis-data.co.uk](http://www.tennis-data.co.uk/)
- Aggregate bookmaker odds or select most popular bookmaker
- Clean and filter to latest Grand Slam tournament
- Convert odds to implied probabilities
- Test two strategies:
  1. **All-in on predicted winner:** Bet on model's highest-probability outcome
  2. **Hedge based on model probabilities:** Size positions according to edge between model and market probabilities
- Evaluate: accuracy vs. odds baseline, ROI, expected value analysis




### References
**model sections**
- Niculescu-Mizil, Alexandru & Caruana, Rich. (2005). Predicting good probabilities with supervised learning. ICML 2005 - Proceedings of the 22nd International Conference on Machine Learning. 625-632. 10.1145/1102351.1102430. 
- Wilson, Ashia & Roelofs, Rebecca & Stern, Mitchell & Srebro, Nathan & Recht, Benjamin. (2017). The Marginal Value of Adaptive Gradient Methods in Machine Learning. 10.48550/arXiv.1705.08292. 
- Hinton, Geoffrey & Srivastava, Nitish & Krizhevsky, Alex & Sutskever, Ilya & Rachmad, Yoesoep & Salakhutdinov, Ruslan. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research. 15. 1929-1958. 
- Guo, Chuan & Pleiss, Geoff & Sun, Yu & Weinberger, Kilian. (2017). On Calibration of Modern Neural Networks. 10.48550/arXiv.1706.04599. 
- Gorishniy, Yury & Rubachev, Ivan & Khrulkov, Valentin & Babenko, Artem. (2021). Revisiting Deep Learning Models for Tabular Data. 10.48550/arXiv.2106.11959. 
- Shmuel, Assaf & Glickman, Oren & Lazebnik, Teddy. (2025). A comprehensive benchmark of machine and deep learning models on structured data for regression and classification. Neurocomputing. 655. 131337. 10.1016/j.neucom.2025.131337. 
- Kingma, Diederik & Ba, Jimmy. (2014). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations. 
