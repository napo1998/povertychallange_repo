# Solution - World Bank Poverty Prediction Challenge 2026 

Dedicated to a person that change my life in the last months.

Building Artificial Intelligence for social good 

**Username:** Napoleon Alcides Perez Arteaga.

University: University of Burgos.

Degree: Doctor of  Computer Science

**LinkedIn:** https://www.linkedin.com/in/napo1998/

---

## Table of Contents

- [Summary](#summary)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Run Training and Inference](#run-training-and-inference)
- [Hardware](#hardware)
- [Approach Details](#approach-details)
  - [Feature Engineering](#feature-engineering)
  - [Model Ensemble](#model-ensemble)
  - [Poverty Rate Estimation](#poverty-rate-estimation)
  - [Validation Strategy](#validation-strategy)

---

## Summary

The solution predicts household-level consumption and poverty rate distributions for the World Bank Poverty Prediction challenge. It produces two outputs:

1. **Household consumption** (`cons_ppp17`): predicted daily per-capita consumption in 2017 PPP dollars for each household.
2. **Poverty rate distributions**: the weighted fraction of households below each of 19 poverty thresholds (5th to 95th percentile in 5-point increments) per survey.

The final competition metric is **0.9 * poverty_wMAPE + 0.1 * consumption_MAPE**, making poverty rate accuracy the dominant component.

### Approach

The solution is a **7-model gradient boosting ensemble** (3 LightGBM, 1 XGBoost, 1 CatBoost, 2 MAPE-weighted LightGBM) trained with **5-fold stratified cross-validation** in log-space (`log1p` of consumption). Key techniques:

- **Extensive feature engineering** (100+ features): household composition ratios, food consumption variety scores (protein/staple/luxury), infrastructure index, employment interactions, survey-relative z-scores and percentile ranks, poverty-line distance ratios, and quadratic/log transforms.
- **Dual ensemble optimization**: separate Nelder-Mead-optimized blend weights for the poverty rate component (total-metric-optimized) and the consumption component (MAPE-optimized), since a single set of weights cannot simultaneously optimize both.
- **Soft-threshold poverty rates**: instead of hard classification (poor/not-poor), each household's probability of being below a poverty line is modeled using the Gaussian CDF over log-space residuals: `P(poor) = Phi(log(threshold / prediction) / sigma)`. Three variants are evaluated: global sigma, per-threshold sigma, and Student-t (heavier tails).
- **Leave-One-Survey-Out (LOSO) validation**: with only 3 training surveys, standard cross-validation is unreliable for poverty rate strategy selection. LOSO holds out one entire survey and learns corrections from the other two, providing a more honest estimate of test-time generalization.
- **Multiplicative bias corrections with shrinkage**: systematic biases in predicted poverty rates are corrected by a multiplicative factor (`actual/predicted` ratio averaged across training surveys). Shrinkage regularizes these corrections toward 1.0 to prevent overfitting on the small number of training surveys.

The pipeline is implemented as a single self-contained Python script (`src/solution.py`).

---

## Directory Structure

```
povertychallange_repo/
│
├── data/                                        # Input data (download from competition)
│   ├── train_hh_features.csv                    #   Training household features (104,234 households)
│   ├── train_hh_gt.csv                          #   Ground truth consumption values
│   ├── train_rates_gt.csv                       #   Ground truth poverty rates per survey
│   ├── test_hh_features.csv                     #   Test household features (103,023 households)
│   ├── feature_descriptions.csv                 #   Metadata for all input features
│   └── feature_value_descriptions.csv           #   Value mappings for categorical features
│
├── predictions/                                 # [Generated] Model output files
│   ├── predicted_household_consumption.csv      #   Predicted consumption per household
│   ├── predicted_poverty_distribution.csv       #   Predicted poverty rates per survey
│   └── submission.zip                           #   Zipped submission for competition upload
│
├── src/                                         # Source code
│   └── solution.py                           #   Main solution script (feature engineering,
│                                                #   model training, ensemble optimization, inference)
│
├── requirements.txt                             # Python dependencies (pinned versions)
├── LICENSE                                      # License file
└── README.md                                    # This file
```

### Input Data (`data/`)

| File | Description |
|------|-------------|
| `train_hh_features.csv` | Input features for 104,234 training households |
| `train_hh_gt.csv` | Ground truth per-capita consumption values |
| `train_rates_gt.csv` | Ground truth poverty rates at different thresholds per survey |
| `test_hh_features.csv` | Input features for 103,023 test households |
| `feature_descriptions.csv` | Metadata describing each feature |
| `feature_value_descriptions.csv` | Categorical value mappings and descriptions |

### Predictions Output (`predictions/`)

Generated after running the solution:

| File | Description |
|------|-------------|
| `predicted_household_consumption.csv` | Columns: `survey_id`, `household_id`, `cons_ppp17` (103,023 rows) |
| `predicted_poverty_distribution.csv` | Columns: `survey_id`, 19 `pct_hh_below_*` rate columns (3 rows, one per test survey) |
| `submission.zip` | Compressed archive of both CSVs for competition upload |

### Source Code

| File | Description |
|------|-------------|
| `src/solution.py` | Main solution script: feature engineering, model training, ensemble optimization, and inference |

---

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

Download the competition data files from the competition page and place them in the `data/` directory.

---

## Run Training and Inference

The solution uses a single script that performs training, validation, optimization, and inference in one execution. There is no separate inference step since model weights are not saved to disk — the script trains all models from scratch and generates predictions in a single run.

```bash
python src/solution.py
```

This will:

1. Load and preprocess the training and test data
2. Engineer ~100+ features from raw household survey data
3. Train 7 gradient boosting models with 5-fold CV (producing out-of-fold predictions)
4. Optimize ensemble blend weights via Nelder-Mead (two sets: total-metric and MAPE)
5. Train an uncertainty model for household-level prediction variance
6. Evaluate and select the best poverty rate estimation strategy via LOSO
7. Generate test predictions and save `submission.zip` to `predictions/`

### Console Output

The script prints progress and validation metrics to stdout, including:

- Per-model out-of-fold MAPE
- Optimal ensemble weights
- LOSO poverty wMAPE for each strategy (global sigma, per-threshold sigma, Student-t)
- Correction factors and shrinkage selection
- Final OOF validation scores vs. leaderboard #1 (3.207)
- Per-survey, per-threshold predicted vs. actual poverty rates

### Reproducibility Notes

- All random seeds are fixed (`RANDOM_SEED = 42`, per-model seeds vary for diversity).
- Results may vary slightly across platforms due to floating-point differences in LightGBM/XGBoost/CatBoost tree construction, but the overall score should be comparable.
- The `DATA_DIR` variable at the top of `src/solution.py` must be updated if running from a different directory.

---

## Hardware

The solution was run on Windows 11.

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5 7640HS w/ Radeon 760M Graphics (6 cores / 12 threads) |
| GPU | N/A (all training and inference on CPU) |
| Memory | 24 GB DDR5 (8 GB + 16 GB) |
| OS | Windows 11 Home |

- **Training time:** ~14 minutes end-to-end (training + optimization + inference) in a single execution.
- All models are trained on CPU using multi-threaded tree building (`n_jobs=-1`).

---

## Approach Details

### Feature Engineering

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

#### Household Labor Features

| Feature | Description |
|---------|-------------|
| `workers_ratio` | Share of household members who are workers |
| `formal_workers_ratio` | Share of household members who are formally employed |
| `formal_of_workers` | Proportion of workers in formal employment (`formal / total workers`). Captures employment quality beyond simple employment status |

#### Utility Expenditure Per-Capita Features

Utility spending (in 2017 PPP dollars) is normalized by different household denominators to capture spending intensity at different scales. A small constant (`+ 0.01`) is added to denominators to avoid division by zero.

| Feature | Description |
|---------|-------------|
| `utl_exp_per_person` | Utility expenditure divided by household size |
| `utl_exp_per_adult` | Utility expenditure divided by number of adults |
| `utl_exp_per_worker` | Utility expenditure divided by (worker share x household size) |

#### Log-Transformed Features

Log transforms (`log1p`) are applied to reduce right-skewness in expenditure and household size distributions, improving model performance on these heavy-tailed variables.

| Feature | Description |
|---------|-------------|
| `log_utl_exp` | `log(1 + utility_expenditure)` |
| `log_utl_per_person` | `log(1 + utility_expenditure_per_person)` |
| `log_hsize` | `log(1 + household_size)` |

#### Employment Interaction Features

These capture the joint effect of being employed with the *type* of employment, distinguishing formal and non-agricultural work.

| Feature | Description |
|---------|-------------|
| `employed_x_formal` | Number employed x formal worker share |
| `employed_x_nonagric` | Number employed x non-agricultural work indicator |

#### Food Consumption Diversity Features

Dietary diversity is a well-known proxy for economic well-being. All columns starting with `consumed` are identified as food items, then grouped by item codes into nutritional categories.

| Feature | Description |
|---------|-------------|
| `food_variety_count` | Total number of distinct food items consumed by the household |
| `food_variety_ratio` | Fraction of all possible food items consumed |
| `protein_variety` | Count of protein-rich items consumed (meat, fish, dairy, eggs — codes 700-2200) |
| `staple_variety` | Count of staple foods consumed (grains, roots — codes 100-1900) |
| `luxury_variety` | Count of luxury/non-essential foods consumed (codes 2600-2700, 4300-4700) |
| `protein_ratio` | Share of consumed items that are protein-rich |
| `luxury_ratio` | Share of consumed items that are luxury foods. Higher values likely correlate with lower poverty |

#### Infrastructure and Education Features

| Feature | Description |
|---------|-------------|
| `infra_score` | Additive index of basic services: water + toilet + sewer + electricity access (0-4 scale) |
| `educ_x_employed` | Secondary education share x employment count |
| `educ_x_urban` | Secondary education share x urban residence indicator |

#### Cross-Feature Interaction Terms

Interaction terms (feature A x feature B) allow the model to capture joint effects that neither feature can represent alone. For example, high utility spending in an urban area signals a different economic status than the same spending in a rural area.

| Feature | Description |
|---------|-------------|
| `utl_x_hsize` | Log utility expenditure x household size |
| `utl_x_food` | Log utility expenditure x food variety count |
| `utl_x_urban` | Utility per person x urban indicator |
| `utl_x_infra` | Utility per person x infrastructure score |
| `food_x_hsize` | Food variety count x household size |
| `food_x_educ` | Food variety count x secondary education share |
| `workers_x_utl` | Worker ratio x utility expenditure per person |

#### Polynomial and Transform Features

Squared terms capture non-linear (e.g., U-shaped or diminishing-return) relationships that linear features cannot model directly.

| Feature | Description |
|---------|-------------|
| `age_sq` | Age squared — captures non-linear age effects on consumption |
| `hsize_sq` | Household size squared — captures economies/diseconomies of scale |
| `log_weight` | `log(1 + survey_weight)` — stabilizes the sampling weight distribution |

### Model Ensemble

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

### Poverty Rate Estimation

The key insight is that the competition metric heavily weights poverty rate accuracy (90%), so directly optimizing consumption MAPE is insufficient. The solution uses **soft thresholds** to convert continuous consumption predictions into smooth poverty rate estimates:

```
P(household_i is below threshold t) = CDF( log(t / predicted_consumption_i) / sigma )
```

Three CDF variants are evaluated via LOSO:
- **Global sigma**: single Gaussian spread across all thresholds
- **Per-threshold sigma**: independent sigma optimized for each of the 19 thresholds
- **Student-t**: heavier-tailed distribution parameterized by degrees of freedom

The best base method is then optionally improved with **multiplicative corrections** (ratio of actual-to-predicted rates on training surveys) regularized by a shrinkage parameter optimized via LOSO.

### Validation Strategy

- **5-fold Stratified CV** (stratified by survey_id): produces out-of-fold consumption predictions used for ensemble weight optimization and uncertainty modeling.
- **Leave-One-Survey-Out (LOSO)**: the primary validation for poverty rate strategy selection. Each of the 3 training surveys (100k, 200k, 300k) is held out in turn, and corrections are learned from the remaining 2. This simulates the test-time scenario of predicting poverty rates for unseen surveys.
