# Solution - World Bank Poverty Prediction Challenge 2026

**Username:** Napoleon  Alcides Perez Arteaga

## Summary

The solution predicts household-level consumption and poverty rate distributions for the World Bank Poverty Prediction challenge. It produces two outputs:

1. **Household consumption** (`cons_ppp17`): predicted daily per-capita consumption in 2017 PPP dollars for each household.
2. **Poverty rate distributions**: the weighted fraction of households below each of 19 poverty thresholds (5th to 95th percentile in 5-point increments) per survey.

The final competition metric is **0.9 * poverty_wMAPE + 0.1 * consumption_MAPE**, making poverty rate accuracy the dominant component.

### Approach

The solution is a **7-model gradient boosting ensemble** (3 LightGBM, 1 XGBoost, 1 CatBoost, 2 MAPE-weighted LightGBM) trained with **5-fold stratified cross-validation** in log-space (`log1p` of consumption). Key techniques:

- **Extensive feature engineering** (~100+ features): household composition ratios, food consumption variety scores (protein/staple/luxury), infrastructure index, employment interactions, survey-relative z-scores and percentile ranks, poverty-line distance ratios, and quadratic/log transforms.
- **Dual ensemble optimization**: separate Nelder-Mead-optimized blend weights for the poverty rate component (total-metric-optimized) and the consumption component (MAPE-optimized), since a single set of weights cannot simultaneously optimize both.
- **Soft-threshold poverty rates**: instead of hard classification (poor/not-poor), each household's probability of being below a poverty line is modeled using the Gaussian CDF over log-space residuals: `P(poor) = Phi(log(threshold / prediction) / sigma)`. Three variants are evaluated: global sigma, per-threshold sigma, and Student-t (heavier tails).
- **Leave-One-Survey-Out (LOSO) validation**: with only 3 training surveys, standard cross-validation is unreliable for poverty rate strategy selection. LOSO holds out one entire survey and learns corrections from the other two, providing a more honest estimate of test-time generalization.
- **Multiplicative bias corrections with shrinkage**: systematic biases in predicted poverty rates are corrected by a multiplicative factor (`actual/predicted` ratio averaged across training surveys). Shrinkage regularizes these corrections toward 1.0 to prevent overfitting on the small number of training surveys.

The pipeline is implemented as a single self-contained Python script (`solution_v6.py`).

## Setup

### Prerequisites

- Python 3.12 (the solution was originally run on Python 3.12.6)

### Create environment

```bash
conda create --name worldbank-poverty python=3.12
conda activate worldbank-poverty
```

Or using `venv`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download data

Download the competition data files from the competition page and place them in the project root directory.

### Directory structure

The structure of the directory before running the solution should be:

```
worldbanksubmission/
├── train_hh_features.csv          <- Training household features (104,234 households)
├── train_hh_gt.csv                <- Training ground truth consumption values
├── train_rates_gt.csv             <- Training ground truth poverty rates per survey
├── test_hh_features.csv           <- Test household features (103,023 households)
├── solution_v6.py                 <- Main solution script (single entry point)
├── requirements.txt               <- Python package dependencies with pinned versions
└── README.md                      <- This file
```

After running, the following output files are generated:

```
worldbanksubmission/
├── predicted_household_consumption.csv   <- Predicted consumption per household
├── predicted_poverty_distribution.csv    <- Predicted poverty rates per survey
├── submission.zip                        <- Zipped submission (both CSVs)
└── catboost_info/                        <- CatBoost training metadata (auto-generated)
```

## Hardware

The solution was run on Windows 11.

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5 7640HS w/ Radeon 760M Graphics (6 cores / 12 threads) |
| GPU | N/A (all training and inference on CPU) |
| Memory | 24 GB DDR5 (8 GB + 16 GB) |
| OS | Windows 11 Home |

- **Training time:** The entire pipeline (training + optimization + inference) runs end-to-end in a single execution.
- All models are trained on CPU using multi-threaded tree building (`n_jobs=-1`).

## Run training and inference

The solution uses a single script that performs training, validation, optimization, and inference in one execution. There is no separate inference step since model weights are not saved to disk -- the script trains all models from scratch and generates predictions in a single run.

```bash
python solution_v6.py
```

This will:

1. Load and preprocess the training and test data
2. Engineer ~100+ features from raw household survey data
3. Train 7 gradient boosting models with 5-fold CV (producing out-of-fold predictions)
4. Optimize ensemble blend weights via Nelder-Mead (two sets: total-metric and MAPE)
5. Train an uncertainty model for household-level prediction variance
6. Evaluate and select the best poverty rate estimation strategy via LOSO
7. Generate test predictions and save `submission.zip`

### Outputs

| File | Description |
|------|-------------|
| `predicted_household_consumption.csv` | Columns: `survey_id`, `household_id`, `cons_ppp17` (103,023 rows) |
| `predicted_poverty_distribution.csv` | Columns: `survey_id`, 19 `pct_hh_below_*` rate columns (3 rows, one per test survey) |
| `submission.zip` | Compressed archive of both CSV files for upload |

### Console output

The script prints progress and validation metrics to stdout, including:

- Per-model out-of-fold MAPE
- Optimal ensemble weights
- LOSO poverty wMAPE for each strategy (global sigma, per-threshold sigma, Student-t)
- Correction factors and shrinkage selection
- Final OOF validation scores vs. leaderboard #1 (3.207)
- Per-survey, per-threshold predicted vs. actual poverty rates

### Reproducibility notes

- All random seeds are fixed (`RANDOM_SEED = 42`, per-model seeds vary for diversity).
- Results may vary slightly across platforms due to floating-point differences in LightGBM/XGBoost/CatBoost tree construction, but the overall score should be comparable.
- The `DATA_DIR` variable at the top of `solution_v6.py` must be updated if running from a different directory.

## Approach details

### Feature engineering

Features are constructed from raw household survey variables and fall into several categories:

| Category | Examples | Count |
|----------|----------|-------|
| Household composition | dependency ratio, children ratio, single adult flag | ~13 |
| Binary encodings | employed, urban, infrastructure access (water/toilet/sewer/electricity) | 9 |
| Economic indicators | utility expenditure per person/adult/worker, formal worker ratio | ~8 |
| Food consumption | variety count, protein/staple/luxury variety, protein ratio | ~6 |
| Interaction terms | education x employment, utility x food, food x infrastructure | ~15 |
| Survey-relative | z-scores and percentile ranks within each survey | ~10 |
| Poverty line distance | utility expenditure / poverty line, log ratios | variable |
| Transforms | log, squared terms | ~5 |

All features are computed on the concatenated train+test data to ensure consistent encoding. Categorical variables are label-encoded. Missing values are filled with -1.

### Model ensemble

Seven gradient boosting models are trained in log-space (`log1p(consumption)`):

| Model | Framework | Objective | Weighting | Key hyperparameters |
|-------|-----------|-----------|-----------|-------------------|
| lgb_huber | LightGBM | Huber | Standard | 800 trees, depth 7, lr 0.04 |
| lgb_mse | LightGBM | MSE | Standard | 700 trees, depth 6, lr 0.04 |
| lgb_mae | LightGBM | MAE | Standard | 600 trees, depth 7, lr 0.05 |
| xgb1 | XGBoost | Squared error | Standard | 600 trees, depth 6, lr 0.05 |
| cat1 | CatBoost | RMSE | Standard | 600 trees, depth 7, lr 0.05 |
| lgb_mape1 | LightGBM | MAE | MAPE-weighted | 600 trees, depth 7, lr 0.05 |
| lgb_mape2 | LightGBM | Huber | MAPE-weighted | 500 trees, depth 6, lr 0.05 |

MAPE-weighted models upweight low-consumption (poor) households by dividing sample weights by consumption value, focusing the loss function on relative prediction accuracy for the poorest households.

Test predictions blend 50% CV-averaged predictions with 50% full-data model predictions.

### Poverty rate estimation

The key insight is that the competition metric heavily weights poverty rate accuracy (90%), so directly optimizing consumption MAPE is insufficient. The solution uses **soft thresholds** to convert continuous consumption predictions into smooth poverty rate estimates:

```
P(household_i is below threshold t) = CDF( log(t / predicted_consumption_i) / sigma )
```

Three CDF variants are evaluated via LOSO:
- **Global sigma**: single Gaussian spread across all thresholds
- **Per-threshold sigma**: independent sigma optimized for each of the 19 thresholds
- **Student-t**: heavier-tailed distribution parameterized by degrees of freedom

The best base method is then optionally improved with **multiplicative corrections** (ratio of actual-to-predicted rates on training surveys) regularized by a shrinkage parameter optimized via LOSO.

### Validation strategy

- **5-fold Stratified CV** (stratified by survey_id): produces out-of-fold consumption predictions used for ensemble weight optimization and uncertainty modeling.
- **Leave-One-Survey-Out (LOSO)**: the primary validation for poverty rate strategy selection. Each of the 3 training surveys (100k, 200k, 300k) is held out in turn, and corrections are learned from the remaining 2. This simulates the test-time scenario of predicting poverty rates for unseen surveys.


