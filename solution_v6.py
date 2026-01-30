"""
World Bank Poverty Prediction - v6  (target: beat #1 = 3.207)
v5 LOSO: 3.19 (poverty=0.61, cons=26.4%)  -- neck and neck with #1

Improvements over v5:
1. 5-fold CV for more stable OOF (80% train vs 67%)
2. More diverse models: +CatBoost, +multiple MAPE-weighted
3. Per-threshold sigma optimization (different sigma per threshold)
4. Correction shrinkage (prevent overfit on 3 training surveys)
5. Student-t soft thresholds (heavier tails than Gaussian)
6. Blended corrections from LOSO

Pipeline Overview
-----------------
The solution predicts two outputs for the World Bank poverty prediction competition:
  1. Household-level consumption (cons_ppp17) in PPP$2017/day
  2. Poverty rate distributions: fraction of households below 19 poverty thresholds
     (5th to 95th percentile in 5-point increments)

The final score is: 0.9 * poverty_wMAPE + 0.1 * consumption_MAPE
Since poverty rates dominate (90%), the pipeline uses separate optimization for each:
  - Consumption predictions use MAPE-optimized ensemble weights
  - Poverty rate predictions use total-metric-optimized weights + soft thresholds

Pipeline Stages:
  PART 1 - Train a diverse regression ensemble (LightGBM, XGBoost, CatBoost) with
           5-fold CV to produce out-of-fold (OOF) predictions. Models are trained
           in log-space (log1p of consumption) to handle skewed distributions.
           Two weighting schemes: standard sample weights and MAPE-weighted.
  PART 2 - Train an uncertainty model that predicts per-household prediction error
           (sigma), used later for soft-threshold poverty rate estimation.
  PART 3 - Evaluate three poverty rate estimation strategies via Leave-One-Survey-Out
           (LOSO) cross-validation on the 3 training surveys (100k, 200k, 300k):
             a) Global sigma: single Gaussian CDF spread parameter across all thresholds
             b) Per-threshold sigma: independent sigma per poverty threshold
             c) Student-t: heavier-tailed distribution for robustness
           Then apply multiplicative corrections (with shrinkage) to reduce bias.
  PART 4 - Select the best strategy based on LOSO scores.
  PART 5 - Full OOF validation and comparison
  PART 6 - Generate predictions for test surveys (400k, 500k, 600k).
  PART 7 - Save submission CSV files and zip archive.

Data Files:
  - train_hh_features.csv:  Household-level features for training surveys
  - train_hh_gt.csv:        Ground truth consumption values per household
  - train_rates_gt.csv:     Ground truth poverty rates per survey
  - test_hh_features.csv:   Household-level features for test surveys
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "c:/Users/napop/OneDrive - Universidad Don Bosco/Escritorio/worldbanksubmission"

# 19 poverty line thresholds in PPP$2017/day, corresponding to the
# 5th, 10th, ..., 95th percentiles of the consumption distribution.
THRESHOLDS = [3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
              9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37]

# Weights for each threshold in the wMAPE calculation.
# Thresholds near the 40th percentile (international poverty line region)
# receive the highest weight (1.0), with linearly decreasing weight
# for thresholds further away. This focuses the metric on the most
# policy-relevant part of the distribution.
THRESHOLD_WEIGHTS = []
for i, t in enumerate(THRESHOLDS):
    percentile = (i + 1) * 5
    weight = 1 - abs(percentile - 40) / 100
    THRESHOLD_WEIGHTS.append(weight)
THRESHOLD_WEIGHTS = np.array(THRESHOLD_WEIGHTS)

N_FOLDS = 5       # Number of CV folds for out-of-fold prediction
RANDOM_SEED = 42

# ================================================================
# DATA LOADING
# ================================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_features = pd.read_csv(f"{DATA_DIR}/train_hh_features.csv")
train_gt = pd.read_csv(f"{DATA_DIR}/train_hh_gt.csv")
train_rates = pd.read_csv(f"{DATA_DIR}/train_rates_gt.csv")
test_features = pd.read_csv(f"{DATA_DIR}/test_hh_features.csv")

print(f"Training: {len(train_features)} | Test: {len(test_features)}")

# ================================================================
# FEATURE ENGINEERING 
# ================================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

id_cols = ['hhid', 'survey_id']
target_col = 'cons_ppp17'
feature_cols = [col for col in train_features.columns if col not in id_cols]
categorical_cols = train_features[feature_cols].select_dtypes(include=['object']).columns.tolist()

all_data = pd.concat([train_features, test_features], ignore_index=True)

print("Creating features...")

# Household composition
all_data['total_children'] = all_data['num_children5'] + all_data['num_children10'] + all_data['num_children18']
all_data['total_adults'] = all_data['num_adult_female'] + all_data['num_adult_male']
all_data['total_dependents'] = all_data['total_children'] + all_data['num_elderly']
all_data['dependency_ratio'] = all_data['total_dependents'] / (all_data['total_adults'] + 0.01)
all_data['children_ratio'] = all_data['total_children'] / (all_data['hsize'] + 0.01)
all_data['adult_ratio'] = all_data['total_adults'] / (all_data['hsize'] + 0.01)
all_data['elderly_ratio'] = all_data['num_elderly'] / (all_data['hsize'] + 0.01)
all_data['female_ratio'] = all_data['num_adult_female'] / (all_data['total_adults'] + 0.01)
all_data['young_children_ratio'] = all_data['num_children5'] / (all_data['hsize'] + 0.01)
all_data['has_elderly'] = (all_data['num_elderly'] > 0).astype(int)
all_data['has_young_children'] = (all_data['num_children5'] > 0).astype(int)
all_data['single_adult'] = (all_data['total_adults'] == 1).astype(int)
all_data['large_household'] = (all_data['hsize'] >= 6).astype(int)

binary_map = {
    'employed': {'Employed': 1, 'Not employed': 0},
    'any_nonagric': {'Yes': 1, 'No': 0},
    'urban': {'Urban': 1, 'Rural': 0},
    'male': {'Male': 1, 'Female': 0},
    'owner': {'Owner': 1, 'Not owner': 0},
    'water': {'Access': 1, 'No access': 0},
    'toilet': {'Access': 1, 'No access': 0},
    'sewer': {'Access': 1, 'No access': 0},
    'elect': {'Access': 1, 'No access': 0},
}
for col, mapping in binary_map.items():
    all_data[f'{col}_num'] = all_data[col].map(mapping).fillna(0).astype(float)

all_data['workers_ratio'] = all_data['sworkershh']
all_data['formal_workers_ratio'] = all_data['sfworkershh']
all_data['formal_of_workers'] = all_data['sfworkershh'] / (all_data['sworkershh'] + 0.01)
all_data['utl_exp_per_person'] = all_data['utl_exp_ppp17'] / (all_data['hsize'] + 0.01)
all_data['utl_exp_per_adult'] = all_data['utl_exp_ppp17'] / (all_data['total_adults'] + 0.01)
all_data['utl_exp_per_worker'] = all_data['utl_exp_ppp17'] / (all_data['sworkershh'] * all_data['hsize'] + 0.01)
all_data['log_utl_exp'] = np.log1p(all_data['utl_exp_ppp17'])
all_data['log_utl_per_person'] = np.log1p(all_data['utl_exp_per_person'])
all_data['log_hsize'] = np.log1p(all_data['hsize'])
all_data['employed_x_formal'] = all_data['employed_num'] * all_data['sfworkershh']
all_data['employed_x_nonagric'] = all_data['employed_num'] * all_data['any_nonagric_num']

food_cols = [col for col in all_data.columns if col.startswith('consumed')]
all_data['food_variety_count'] = all_data[food_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
all_data['food_variety_ratio'] = all_data['food_variety_count'] / len(food_cols)

protein_cols = [c for c in food_cols if any(x in c for x in ['800', '900', '1000', '1100', '1200', '1300', '1400', '2000', '2100', '2200', '700'])]
staple_cols = [c for c in food_cols if any(x in c for x in ['100', '300', '500', '1500', '1600', '1900'])]
luxury_cols = [c for c in food_cols if any(x in c for x in ['4300', '4400', '4500', '4600', '4700', '2600', '2700'])]

all_data['protein_variety'] = all_data[protein_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
all_data['staple_variety'] = all_data[staple_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
all_data['luxury_variety'] = all_data[luxury_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
all_data['protein_ratio'] = all_data['protein_variety'] / (all_data['food_variety_count'] + 0.01)
all_data['luxury_ratio'] = all_data['luxury_variety'] / (all_data['food_variety_count'] + 0.01)

all_data['infra_score'] = all_data['water_num'] + all_data['toilet_num'] + all_data['sewer_num'] + all_data['elect_num']
all_data['educ_x_employed'] = all_data['share_secondary'] * all_data['employed_num']
all_data['educ_x_urban'] = all_data['share_secondary'] * all_data['urban_num']

all_data['utl_x_hsize'] = all_data['log_utl_exp'] * all_data['hsize']
all_data['utl_x_food'] = all_data['log_utl_exp'] * all_data['food_variety_count']
all_data['utl_x_urban'] = all_data['utl_exp_per_person'] * all_data['urban_num']
all_data['utl_x_infra'] = all_data['utl_exp_per_person'] * all_data['infra_score']
all_data['food_x_hsize'] = all_data['food_variety_count'] * all_data['hsize']
all_data['food_x_educ'] = all_data['food_variety_count'] * all_data['share_secondary']
all_data['workers_x_utl'] = all_data['workers_ratio'] * all_data['utl_exp_per_person']
all_data['age_sq'] = all_data['age'] ** 2
all_data['hsize_sq'] = all_data['hsize'] ** 2
all_data['log_weight'] = np.log1p(all_data['weight'])

# Survey-relative features
survey_rel_cols = ['utl_exp_ppp17', 'hsize', 'age', 'share_secondary',
                   'food_variety_count', 'sworkershh', 'sfworkershh']
for col in survey_rel_cols:
    s_mean = all_data.groupby('survey_id')[col].transform('mean')
    s_std = all_data.groupby('survey_id')[col].transform('std').replace(0, 1)
    all_data[f'{col}_srel'] = (all_data[col] - s_mean) / s_std

rank_cols = ['utl_exp_ppp17', 'food_variety_count', 'hsize']
for col in rank_cols:
    all_data[f'{col}_srank'] = all_data.groupby('survey_id')[col].rank(pct=True)

# Poverty line distance features
pline_cols = [c for c in all_data.columns if c.startswith('_pline')]
for pcol in pline_cols:
    all_data[f'utl_over{pcol}'] = all_data['utl_exp_ppp17'] / (all_data[pcol] + 0.01)
    all_data[f'logutl_over{pcol}'] = np.log1p(all_data['utl_exp_ppp17']) - np.log1p(all_data[pcol])

# NEW v6: Additional interaction features
all_data['food_x_infra'] = all_data['food_variety_count'] * all_data['infra_score']
all_data['food_x_urban'] = all_data['food_variety_count'] * all_data['urban_num']
all_data['educ_x_infra'] = all_data['share_secondary'] * all_data['infra_score']
all_data['protein_x_utl'] = all_data['protein_variety'] * all_data['log_utl_exp']
all_data['luxury_x_urban'] = all_data['luxury_variety'] * all_data['urban_num']
all_data['hsize_x_urban'] = all_data['hsize'] * all_data['urban_num']
all_data['formal_x_urban'] = all_data['formal_workers_ratio'] * all_data['urban_num']
all_data['age_x_employed'] = all_data['age'] * all_data['employed_num']

binary_num_features = [f'{col}_num' for col in binary_map.keys()]
survey_rel_features = [f'{col}_srel' for col in survey_rel_cols]
survey_rank_features = [f'{col}_srank' for col in rank_cols]
pline_feat_names = []
for pcol in pline_cols:
    pline_feat_names.append(f'utl_over{pcol}')
    pline_feat_names.append(f'logutl_over{pcol}')

new_features = binary_num_features + [
    'total_children', 'total_adults', 'total_dependents', 'dependency_ratio',
    'children_ratio', 'adult_ratio', 'elderly_ratio', 'female_ratio',
    'young_children_ratio', 'has_elderly', 'has_young_children', 'single_adult', 'large_household',
    'workers_ratio', 'formal_workers_ratio', 'formal_of_workers',
    'utl_exp_per_person', 'utl_exp_per_adult', 'utl_exp_per_worker',
    'log_utl_exp', 'log_utl_per_person', 'log_hsize',
    'employed_x_formal', 'employed_x_nonagric',
    'food_variety_count', 'food_variety_ratio',
    'protein_variety', 'staple_variety', 'luxury_variety', 'protein_ratio', 'luxury_ratio',
    'infra_score', 'educ_x_employed', 'educ_x_urban',
    'utl_x_hsize', 'utl_x_food', 'utl_x_urban', 'utl_x_infra',
    'food_x_hsize', 'food_x_educ', 'workers_x_utl',
    'age_sq', 'hsize_sq', 'log_weight',
    'food_x_infra', 'food_x_urban', 'educ_x_infra',
    'protein_x_utl', 'luxury_x_urban', 'hsize_x_urban',
    'formal_x_urban', 'age_x_employed',
] + survey_rel_features + survey_rank_features + pline_feat_names
feature_cols = feature_cols + new_features

# Encode categoricals
for col in categorical_cols:
    le = LabelEncoder()
    all_data[col] = all_data[col].fillna('missing')
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# Split
train_encoded = all_data.iloc[:len(train_features)].copy()
test_encoded = all_data.iloc[len(train_features):].copy()
train_encoded = train_encoded.merge(train_gt, on=['survey_id', 'hhid'])

train_encoded[feature_cols] = train_encoded[feature_cols].fillna(-1)
test_encoded[feature_cols] = test_encoded[feature_cols].fillna(-1)

X_train = train_encoded[feature_cols].values.astype(np.float32)
y_train = train_encoded[target_col].values
weights_train = train_encoded['weight'].values
survey_ids_train = train_encoded['survey_id'].values

# Normalize survey weights to mean=1 so they act as relative importance
# without changing the effective sample size for the loss function
weights_norm = weights_train / weights_train.sum() * len(weights_train)
# Models are trained in log-space to handle the right-skewed consumption distribution
y_train_log = np.log1p(y_train)

X_test = test_encoded[feature_cols].values.astype(np.float32)
test_weights = test_encoded['weight'].values
test_survey_ids = test_encoded['survey_id'].values

print(f"Total features: {len(feature_cols)}")

# ================================================================
# HELPERS
# ================================================================

def calculate_weighted_poverty_rates(consumption, weights, thresholds):
    """Compute the weighted fraction of households below each poverty threshold.

    For each threshold t, calculates:
        rate_t = sum(weights[consumption < t]) / sum(weights)

    Args:
        consumption: Array of household consumption values (PPP$/day).
        weights: Array of survey sampling weights per household.
        thresholds: List of poverty line values to evaluate.

    Returns:
        Dict mapping 'pct_hh_below_{threshold}' to the weighted poverty rate.
    """
    total_weight = weights.sum()
    rates = {}
    for t in thresholds:
        rates[f'pct_hh_below_{t:.2f}'] = weights[consumption < t].sum() / total_weight
    return rates

def calculate_wmape(pred_rates, actual_rates, thresholds, threshold_weights):
    """Compute the weighted Mean Absolute Percentage Error (wMAPE) across thresholds.

    For each threshold, computes APE = |predicted - actual| / actual, then
    returns the weighted average across thresholds (scaled to percentage).
    Thresholds with actual_rate == 0 are skipped.

    Args:
        pred_rates: Dict of predicted poverty rates by threshold key.
        actual_rates: Dict of actual poverty rates by threshold key.
        thresholds: List of poverty line values.
        threshold_weights: Array of importance weights per threshold.

    Returns:
        Weighted MAPE as a percentage (0-100 scale).
    """
    errors = []
    weights_sum = 0
    for i, t in enumerate(thresholds):
        key = f'pct_hh_below_{t:.2f}'
        if actual_rates[key] > 0:
            err = abs(pred_rates[key] - actual_rates[key]) / actual_rates[key]
            errors.append(err * threshold_weights[i])
            weights_sum += threshold_weights[i]
    if weights_sum == 0:
        return 0
    return sum(errors) / weights_sum * 100

def eval_metric(pred_cons, y_true, w_train, survey_ids):
    """Compute the full competition metric: 0.9 * poverty_wMAPE + 0.1 * consumption_MAPE.

    Poverty wMAPE is averaged across surveys (each survey contributes equally).
    Consumption MAPE is computed globally across all households.

    Args:
        pred_cons: Array of predicted consumption values.
        y_true: Array of actual consumption values.
        w_train: Array of survey sampling weights.
        survey_ids: Array of survey identifiers per household.

    Returns:
        Tuple of (total_score, poverty_wMAPE, consumption_MAPE).
    """
    rate_errors = []
    for sid in np.unique(survey_ids):
        mask = survey_ids == sid
        pr = calculate_weighted_poverty_rates(pred_cons[mask], w_train[mask], THRESHOLDS)
        ar = calculate_weighted_poverty_rates(y_true[mask], w_train[mask], THRESHOLDS)
        rate_errors.append(calculate_wmape(pr, ar, THRESHOLDS, THRESHOLD_WEIGHTS))
    pov = np.mean(rate_errors)
    cmape = np.mean(np.abs(pred_cons - y_true) / y_true) * 100
    return 0.9 * pov + 0.1 * cmape, pov, cmape

def eval_poverty_only(pred_rates_by_survey, actual_rates_by_survey):
    """Compute mean poverty wMAPE across surveys from pre-computed rate dicts.

    Unlike eval_metric, this works with pre-computed poverty rate dictionaries
    (e.g., from soft-threshold methods) rather than raw consumption predictions.

    Args:
        pred_rates_by_survey: Dict mapping survey_id -> rate dict.
        actual_rates_by_survey: Dict mapping survey_id -> rate dict.

    Returns:
        Mean wMAPE across all surveys (percentage).
    """
    rate_errors = []
    for sid in pred_rates_by_survey:
        wmape = calculate_wmape(pred_rates_by_survey[sid], actual_rates_by_survey[sid],
                                THRESHOLDS, THRESHOLD_WEIGHTS)
        rate_errors.append(wmape)
    return np.mean(rate_errors)

def train_model(model_type, params, X_tr, y_tr, w_tr):
    """Train a gradient boosting model (LightGBM, XGBoost, or CatBoost).

    Factory function that instantiates and fits the appropriate model based
    on the model_type string.

    Args:
        model_type: One of 'lgb', 'xgb', or 'cat'.
        params: Dict of hyperparameters passed to the model constructor.
        X_tr: Training feature matrix.
        y_tr: Training target array (log-transformed consumption).
        w_tr: Sample weights for training.

    Returns:
        Fitted model object.
    """
    if model_type == 'lgb':
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_type == 'cat':
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    return model

# ================================================================
# PART 1: DIVERSE REGRESSION ENSEMBLE
# ================================================================
print("\n" + "=" * 60)
print("PART 1: REGRESSION ENSEMBLE")
print("=" * 60)

model_configs = {
    'lgb_huber': {
        'type': 'lgb',
        'params': {
            'objective': 'huber', 'metric': 'mae',
            'n_estimators': 800, 'max_depth': 7, 'learning_rate': 0.04,
            'num_leaves': 80, 'min_child_samples': 20,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
    },
    'lgb_mse': {
        'type': 'lgb',
        'params': {
            'objective': 'regression', 'metric': 'mae',
            'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.04,
            'num_leaves': 63, 'min_child_samples': 30,
            'subsample': 0.85, 'colsample_bytree': 0.75,
            'reg_alpha': 0.05, 'reg_lambda': 0.2,
            'random_state': 123, 'verbose': -1, 'n_jobs': -1
        }
    },
    'lgb_mae': {
        'type': 'lgb',
        'params': {
            'objective': 'regression_l1', 'metric': 'mae',
            'n_estimators': 600, 'max_depth': 7, 'learning_rate': 0.05,
            'num_leaves': 80, 'min_child_samples': 25,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 456, 'verbose': -1, 'n_jobs': -1
        }
    },
    'xgb1': {
        'type': 'xgb',
        'params': {
            'objective': 'reg:squarederror',
            'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            'tree_method': 'hist'
        }
    },
    'cat1': {
        'type': 'cat',
        'params': {
            'iterations': 600, 'depth': 7, 'learning_rate': 0.05,
            'l2_leaf_reg': 3, 'random_seed': 42,
            'verbose': 0, 'loss_function': 'RMSE'
        }
    },
}

# MAPE-weighted models: upweight low-consumption households by dividing by
# consumption value. This makes the model focus more on getting poor households
# right, since MAPE penalizes relative errors equally across income levels.
mape_weight = weights_norm / np.maximum(y_train, 1.0)
mape_weight = mape_weight / mape_weight.sum() * len(mape_weight)

mape_model_configs = {
    'lgb_mape1': {
        'type': 'lgb',
        'params': {
            'objective': 'regression_l1', 'metric': 'mae',
            'n_estimators': 600, 'max_depth': 7, 'learning_rate': 0.05,
            'num_leaves': 80, 'min_child_samples': 25,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 789, 'verbose': -1, 'n_jobs': -1
        }
    },
    'lgb_mape2': {
        'type': 'lgb',
        'params': {
            'objective': 'huber', 'metric': 'mae',
            'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
            'num_leaves': 63, 'min_child_samples': 30,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 0.15,
            'random_state': 321, 'verbose': -1, 'n_jobs': -1
        }
    },
}

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
folds = list(skf.split(X_train, survey_ids_train))

oof_preds = {}
test_preds = {}

# Train standard models with 5-fold CV.
# OOF predictions are used for ensemble weight optimization (unbiased estimates).
# Test predictions blend 50% fold-averaged CV predictions with 50% full-data model
# predictions to get the stability of CV averaging plus the accuracy of using all data.
for model_name, config in model_configs.items():
    print(f"  Training {model_name}...")
    model_type = config['type']
    params = config['params']

    oof = np.zeros(len(X_train))
    test_pred_folds = np.zeros((N_FOLDS, len(X_test)))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model = train_model(model_type, params, X_train[train_idx],
                           y_train_log[train_idx], weights_norm[train_idx])
        oof[val_idx] = model.predict(X_train[val_idx])
        test_pred_folds[fold_idx] = model.predict(X_test)

    oof_preds[model_name] = oof
    test_preds[model_name] = test_pred_folds.mean(axis=0)

    full_model = train_model(model_type, params, X_train, y_train_log, weights_norm)
    test_preds[model_name] = 0.5 * test_preds[model_name] + 0.5 * full_model.predict(X_test)

    oof_cons = np.expm1(oof)
    mape_val = np.mean(np.abs(oof_cons - y_train) / y_train) * 100
    print(f"    OOF MAPE: {mape_val:.2f}%")

# Train MAPE-weighted models
for model_name, config in mape_model_configs.items():
    print(f"  Training {model_name} (MAPE-weighted)...")
    model_type = config['type']
    params = config['params']

    oof = np.zeros(len(X_train))
    test_pred_folds = np.zeros((N_FOLDS, len(X_test)))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model = train_model(model_type, params, X_train[train_idx],
                           y_train_log[train_idx], mape_weight[train_idx])
        oof[val_idx] = model.predict(X_train[val_idx])
        test_pred_folds[fold_idx] = model.predict(X_test)

    oof_preds[model_name] = oof
    test_preds[model_name] = test_pred_folds.mean(axis=0)

    full_model = train_model(model_type, params, X_train, y_train_log, mape_weight)
    test_preds[model_name] = 0.5 * test_preds[model_name] + 0.5 * full_model.predict(X_test)

    oof_cons = np.expm1(oof)
    mape_val = np.mean(np.abs(oof_cons - y_train) / y_train) * 100
    print(f"    OOF MAPE: {mape_val:.2f}%")

# ================================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ================================================================
print("\n" + "=" * 60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 60)

model_names = list(oof_preds.keys())
n_models = len(model_names)

oof_cons_all = np.column_stack([np.expm1(oof_preds[name]) for name in model_names])
test_cons_all = np.column_stack([np.expm1(test_preds[name]) for name in model_names])

def compute_oof_metric(weights):
    """Objective function for Nelder-Mead: compute the full competition metric on OOF predictions.

    Takes raw ensemble weights (may be negative from optimizer), normalizes them
    to sum to 1, blends OOF predictions from all models, and evaluates the full
    0.9*poverty + 0.1*consumption metric. Used to find optimal weights for
    poverty rate prediction.

    Args:
        weights: Array of model blend weights (will be abs'd and normalized).

    Returns:
        Total competition metric score (lower is better).
    """
    weights = np.abs(weights)
    weights = weights / weights.sum()
    pred = oof_cons_all @ weights
    total, pov, cmape = eval_metric(pred, y_train, weights_train, survey_ids_train)
    return total

def compute_oof_mape(weights):
    """Objective function for Nelder-Mead: compute consumption MAPE only on OOF predictions.

    Similar to compute_oof_metric but optimizes only for household-level consumption
    accuracy (MAPE), ignoring poverty rate accuracy. Used to find optimal weights
    for the consumption submission component.

    Args:
        weights: Array of model blend weights (will be abs'd and normalized).

    Returns:
        Consumption MAPE as a percentage (lower is better).
    """
    weights = np.abs(weights)
    weights = weights / weights.sum()
    pred = oof_cons_all @ weights
    return np.mean(np.abs(pred - y_train) / y_train) * 100

init_weights = np.ones(n_models) / n_models

# Optimize for total metric
result_total = minimize(compute_oof_metric, init_weights, method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
w_total = np.abs(result_total.x)
w_total = w_total / w_total.sum()

# Optimize for MAPE only
result_mape = minimize(compute_oof_mape, init_weights, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
w_mape = np.abs(result_mape.x)
w_mape = w_mape / w_mape.sum()

# Compare
total_score_total = compute_oof_metric(w_total)
total_score_mape = compute_oof_metric(w_mape)
mape_score_total = compute_oof_mape(w_total)
mape_score_mape = compute_oof_mape(w_mape)

print(f"\nTotal-opt weights: total={total_score_total:.4f}, mape={mape_score_total:.2f}%")
print(f"MAPE-opt weights:  total={total_score_mape:.4f}, mape={mape_score_mape:.2f}%")

# Two separate weight sets: the total-metric-optimized weights produce better
# poverty rates (90% of score), while MAPE-optimized weights minimize the
# consumption error component (10% of score). Using separate weights for each
# output avoids the compromise of a single set.
final_weights = w_total  # for soft threshold poverty rates
cons_weights = w_mape    # for consumption submission

print("\nTotal-opt weights:")
for name, w in zip(model_names, final_weights):
    print(f"  {name}: {w:.4f}")
print("MAPE-opt weights:")
for name, w in zip(model_names, cons_weights):
    print(f"  {name}: {w:.4f}")

oof_ensemble = oof_cons_all @ final_weights   # for poverty
oof_cons_best = oof_cons_all @ cons_weights   # for consumption MAPE
test_ensemble = test_cons_all @ final_weights
test_cons_best = test_cons_all @ cons_weights

# ================================================================
# PART 2: HOUSEHOLD-SPECIFIC UNCERTAINTY
# ================================================================
print("\n" + "=" * 60)
print("PART 2: UNCERTAINTY ESTIMATION")
print("=" * 60)

# Compute log-space residuals between actual and predicted consumption.
# The std of these residuals (global_sigma) provides the baseline spread
# parameter for soft-threshold poverty rate estimation.
oof_log_residuals = np.log1p(y_train) - np.log1p(np.maximum(oof_ensemble, 0.01))
abs_residuals = np.abs(oof_log_residuals)
global_sigma = np.std(oof_log_residuals)

print(f"  Global log-residual std: {global_sigma:.4f}")

# Train a LightGBM model to predict per-household |residual| from features.
# This captures heteroscedasticity: some households are inherently harder to
# predict (e.g., informal sector workers with volatile income).
unc_model = lgb.LGBMRegressor(
    objective='regression_l1', n_estimators=300, max_depth=5,
    learning_rate=0.08, num_leaves=31, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.6, verbose=-1, n_jobs=-1, random_state=42
)

oof_sigma = np.full(len(X_train), global_sigma)
test_sigma_folds = np.zeros((N_FOLDS, len(X_test)))

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    unc_model.fit(X_train[train_idx], abs_residuals[train_idx],
                  sample_weight=weights_norm[train_idx])
    oof_sigma[val_idx] = unc_model.predict(X_train[val_idx])
    test_sigma_folds[fold_idx] = unc_model.predict(X_test)

test_sigma = test_sigma_folds.mean(axis=0)
oof_sigma = np.clip(oof_sigma, global_sigma * 0.3, global_sigma * 3.0)
test_sigma = np.clip(test_sigma, global_sigma * 0.3, global_sigma * 3.0)

print(f"  Predicted sigma: mean={oof_sigma.mean():.4f}, std={oof_sigma.std():.4f}")

# ================================================================
# PART 3: POVERTY RATE STRATEGIES
# ================================================================
print("\n" + "=" * 60)
print("PART 3: POVERTY RATE STRATEGIES")
print("=" * 60)

unique_surveys = [100000, 200000, 300000]

actual_rates_by_survey = {}
for sid in unique_surveys:
    mask = survey_ids_train == sid
    actual_rates_by_survey[sid] = calculate_weighted_poverty_rates(
        y_train[mask], weights_train[mask], THRESHOLDS)

# --- Soft thresholds (Gaussian) ---
def soft_rates_global(cons_pred, weights, survey_ids, targets, sigma):
    """Estimate poverty rates using Gaussian soft thresholds with a single global sigma.

    Instead of hard-classifying households as poor/non-poor, this method models
    each household's true consumption as a log-normal random variable centered on
    the predicted value with spread sigma. The probability of being below threshold t
    for household i is:
        P(poor_i | t) = Phi( log(t / cons_pred_i) / sigma )

    where Phi is the standard normal CDF. The poverty rate is the weighted average
    of these probabilities. This smoothing reduces sensitivity to individual
    prediction errors and produces more stable aggregate poverty rates.

    Args:
        cons_pred: Array of predicted consumption values (natural scale).
        weights: Array of survey sampling weights.
        survey_ids: Array of survey identifiers.
        targets: List of survey IDs to compute rates for.
        sigma: Global spread parameter (std of log-residuals).

    Returns:
        Dict mapping survey_id -> rate dict with keys 'pct_hh_below_{threshold}'.
    """
    rates = {}
    for sid in targets:
        mask = survey_ids == sid
        r = {}
        for t in THRESHOLDS:
            log_ratio = np.log(t / np.maximum(cons_pred[mask], 0.01))
            probs = norm.cdf(log_ratio / sigma)
            r[f'pct_hh_below_{t:.2f}'] = np.average(probs, weights=weights[mask])
        rates[sid] = r
    return rates

# --- Soft thresholds (Student-t) for heavier tails ---
def soft_rates_student(cons_pred, weights, survey_ids, targets, sigma, df):
    """Estimate poverty rates using Student-t soft thresholds.

    Same concept as soft_rates_global but uses the Student-t CDF instead of
    the Gaussian. The heavier tails of the Student-t distribution assign more
    probability mass to extreme deviations, which can better account for
    outlier consumption predictions and skewed error distributions.

    Args:
        cons_pred: Array of predicted consumption values (natural scale).
        weights: Array of survey sampling weights.
        survey_ids: Array of survey identifiers.
        targets: List of survey IDs to compute rates for.
        sigma: Spread parameter (scale of log-residuals).
        df: Degrees of freedom for the Student-t distribution. Lower values
            produce heavier tails (df=3 is very heavy; df=50 approaches Gaussian).

    Returns:
        Dict mapping survey_id -> rate dict.
    """
    rates = {}
    for sid in targets:
        mask = survey_ids == sid
        r = {}
        for t in THRESHOLDS:
            log_ratio = np.log(t / np.maximum(cons_pred[mask], 0.01))
            probs = student_t.cdf(log_ratio / sigma, df=df)
            r[f'pct_hh_below_{t:.2f}'] = np.average(probs, weights=weights[mask])
        rates[sid] = r
    return rates

# --- Per-threshold sigma ---
def soft_rates_perthreshold(cons_pred, weights, survey_ids, targets, sigmas):
    """Estimate poverty rates using Gaussian soft thresholds with per-threshold sigma.

    Like soft_rates_global, but uses a different sigma for each poverty threshold.
    This allows the model to capture the fact that prediction uncertainty may affect
    poverty rate estimation differently at different points in the distribution
    (e.g., the tails vs. the center).

    Args:
        cons_pred: Array of predicted consumption values (natural scale).
        weights: Array of survey sampling weights.
        survey_ids: Array of survey identifiers.
        targets: List of survey IDs to compute rates for.
        sigmas: Array of sigma values, one per threshold (length = len(THRESHOLDS)).

    Returns:
        Dict mapping survey_id -> rate dict.
    """
    rates = {}
    for sid in targets:
        mask = survey_ids == sid
        r = {}
        for k, t in enumerate(THRESHOLDS):
            log_ratio = np.log(t / np.maximum(cons_pred[mask], 0.01))
            probs = norm.cdf(log_ratio / sigmas[k])
            r[f'pct_hh_below_{t:.2f}'] = np.average(probs, weights=weights[mask])
        rates[sid] = r
    return rates

# --- Corrected rates ---
def apply_corrections(rates, corrections, shrinkage=1.0):
    """Apply multiplicative bias corrections to predicted poverty rates with shrinkage.

    Each threshold's predicted rate is multiplied by a correction factor learned
    from training data (actual/predicted ratio). Shrinkage regularizes the
    corrections toward 1.0 (no correction) to prevent overfitting when corrections
    are learned from only a few surveys.

    The adjusted correction is: adj = 1.0 + shrinkage * (raw_correction - 1.0)
    So shrinkage=0 means no correction, shrinkage=1 means full correction.

    Args:
        rates: Dict mapping survey_id -> rate dict (predicted rates).
        corrections: Array of multiplicative correction factors per threshold.
        shrinkage: Float in [0, 1] controlling how much correction to apply.
            0.0 = no correction, 1.0 = full correction.

    Returns:
        Dict mapping survey_id -> corrected rate dict (clipped to [0, 1]).
    """
    corrected = {}
    for sid in rates:
        r = {}
        for k, t in enumerate(THRESHOLDS):
            key = f'pct_hh_below_{t:.2f}'
            adj_corr = 1.0 + shrinkage * (corrections[k] - 1.0)
            r[key] = np.clip(rates[sid][key] * adj_corr, 0, 1)
        corrected[sid] = r
    return corrected

# ================================================================
# PART 3a: Find best base sigma
# ================================================================
print("\n  Finding optimal global sigma...")

best_global_sigma = None
best_global_score = float('inf')
for ratio in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]:
    sigma = global_sigma * ratio
    loso_errors = []
    for held_out in unique_surveys:
        pred_rates = soft_rates_global(oof_ensemble, weights_train, survey_ids_train,
                                       [held_out], sigma)
        wmape = calculate_wmape(pred_rates[held_out], actual_rates_by_survey[held_out],
                                THRESHOLDS, THRESHOLD_WEIGHTS)
        loso_errors.append(wmape)
    avg = np.mean(loso_errors)
    if avg < best_global_score:
        best_global_score = avg
        best_global_sigma = sigma

# Fine-tune
for delta in np.arange(-0.03, 0.035, 0.002):
    sigma = best_global_sigma + delta
    if sigma <= 0:
        continue
    loso_errors = []
    for held_out in unique_surveys:
        pred_rates = soft_rates_global(oof_ensemble, weights_train, survey_ids_train,
                                       [held_out], sigma)
        wmape = calculate_wmape(pred_rates[held_out], actual_rates_by_survey[held_out],
                                THRESHOLDS, THRESHOLD_WEIGHTS)
        loso_errors.append(wmape)
    avg = np.mean(loso_errors)
    if avg < best_global_score:
        best_global_score = avg
        best_global_sigma = sigma

print(f"  Best global sigma: {best_global_sigma:.4f} (ratio={best_global_sigma/global_sigma:.3f}), "
      f"LOSO={best_global_score:.4f}")

# ================================================================
# PART 3b: Per-threshold sigma optimization
# ================================================================
print("\n  Optimizing per-threshold sigma...")

per_t_sigmas = np.full(len(THRESHOLDS), best_global_sigma)
# Optimize each threshold's sigma independently
for k, t in enumerate(THRESHOLDS):
    key = f'pct_hh_below_{t:.2f}'
    best_s = best_global_sigma
    best_err = float('inf')
    for ratio in np.arange(0.6, 1.5, 0.05):
        s = best_global_sigma * ratio
        loso_errs = []
        for held_out in unique_surveys:
            mask = survey_ids_train == held_out
            log_ratio = np.log(t / np.maximum(oof_ensemble[mask], 0.01))
            probs = norm.cdf(log_ratio / s)
            pred_rate = np.average(probs, weights=weights_train[mask])
            actual_rate = actual_rates_by_survey[held_out][key]
            if actual_rate > 0:
                loso_errs.append(abs(pred_rate - actual_rate) / actual_rate)
        avg_err = np.mean(loso_errs) if loso_errs else float('inf')
        if avg_err < best_err:
            best_err = avg_err
            best_s = s
    per_t_sigmas[k] = best_s

# Evaluate per-threshold sigma
loso_errors = []
for held_out in unique_surveys:
    pred_rates = soft_rates_perthreshold(oof_ensemble, weights_train, survey_ids_train,
                                          [held_out], per_t_sigmas)
    wmape = calculate_wmape(pred_rates[held_out], actual_rates_by_survey[held_out],
                            THRESHOLDS, THRESHOLD_WEIGHTS)
    loso_errors.append(wmape)
per_t_loso = np.mean(loso_errors)
print(f"  Per-threshold sigma LOSO: {per_t_loso:.4f}")

# Show sigma ratios
for k in [0, 3, 7, 10, 14, 18]:
    pct = (k + 1) * 5
    print(f"    {pct}th: sigma={per_t_sigmas[k]:.4f} (ratio={per_t_sigmas[k]/global_sigma:.3f})")

# ================================================================
# PART 3c: Student-t soft thresholds
# ================================================================
print("\n  Testing Student-t distribution...")

best_student_df = None
best_student_sigma = None
best_student_score = float('inf')
for df_val in [3, 5, 8, 12, 20, 50]:
    for ratio in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        sigma = global_sigma * ratio
        loso_errors = []
        for held_out in unique_surveys:
            pred_rates = soft_rates_student(oof_ensemble, weights_train, survey_ids_train,
                                             [held_out], sigma, df_val)
            wmape = calculate_wmape(pred_rates[held_out], actual_rates_by_survey[held_out],
                                    THRESHOLDS, THRESHOLD_WEIGHTS)
            loso_errors.append(wmape)
        avg = np.mean(loso_errors)
        if avg < best_student_score:
            best_student_score = avg
            best_student_df = df_val
            best_student_sigma = sigma

print(f"  Student-t: df={best_student_df}, sigma={best_student_sigma:.4f}, "
      f"LOSO={best_student_score:.4f}")

# ================================================================
# PART 3d: Corrections with shrinkage (LOSO)
# ================================================================
print("\n  Computing corrections with LOSO shrinkage...")

# Choose best base approach
base_approaches = {
    'global': (best_global_score, 'global'),
    'per_t': (per_t_loso, 'per_t'),
    'student': (best_student_score, 'student'),
}
best_base_name = min(base_approaches, key=lambda x: base_approaches[x][0])
best_base_score = base_approaches[best_base_name][0]
print(f"  Best base: {best_base_name} (LOSO={best_base_score:.4f})")

def get_base_rates(cons_pred, weights, survey_ids, targets):
    """Dispatch to the best-performing soft-threshold poverty rate method.

    Routes to the soft-threshold function that achieved the lowest LOSO
    poverty wMAPE during Part 3 evaluation (global, per-threshold, or Student-t),
    using the optimized parameters found for that method.

    Args:
        cons_pred: Array of predicted consumption values (natural scale).
        weights: Array of survey sampling weights.
        survey_ids: Array of survey identifiers.
        targets: List of survey IDs to compute rates for.

    Returns:
        Dict mapping survey_id -> rate dict.
    """
    if best_base_name == 'global':
        return soft_rates_global(cons_pred, weights, survey_ids, targets, best_global_sigma)
    elif best_base_name == 'per_t':
        return soft_rates_perthreshold(cons_pred, weights, survey_ids, targets, per_t_sigmas)
    elif best_base_name == 'student':
        return soft_rates_student(cons_pred, weights, survey_ids, targets,
                                  best_student_sigma, best_student_df)

# LOSO correction evaluation with different shrinkage levels
best_shrinkage = 1.0
best_corr_score = float('inf')

for shrinkage in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    loso_errors = []
    for held_out in unique_surveys:
        other_surveys = [s for s in unique_surveys if s != held_out]

        # Learn corrections from OTHER surveys only
        corrections = np.ones(len(THRESHOLDS))
        for k, t in enumerate(THRESHOLDS):
            key = f'pct_hh_below_{t:.2f}'
            pred_sum = 0
            actual_sum = 0
            for os in other_surveys:
                base_rates = get_base_rates(oof_ensemble, weights_train, survey_ids_train, [os])
                pred_sum += base_rates[os][key]
                actual_sum += actual_rates_by_survey[os][key]
            if pred_sum > 0:
                corrections[k] = actual_sum / pred_sum

        # Apply with shrinkage
        pred_rates = get_base_rates(oof_ensemble, weights_train, survey_ids_train, [held_out])
        pred_rates = apply_corrections(pred_rates, corrections, shrinkage)
        wmape = calculate_wmape(pred_rates[held_out], actual_rates_by_survey[held_out],
                                THRESHOLDS, THRESHOLD_WEIGHTS)
        loso_errors.append(wmape)

    avg = np.mean(loso_errors)
    if avg < best_corr_score:
        best_corr_score = avg
        best_shrinkage = shrinkage

print(f"  Best shrinkage={best_shrinkage:.1f}: LOSO corrected = {best_corr_score:.4f}")

# Learn final corrections from ALL surveys
final_corrections = np.ones(len(THRESHOLDS))
for k, t in enumerate(THRESHOLDS):
    key = f'pct_hh_below_{t:.2f}'
    pred_sum = 0
    actual_sum = 0
    for sid in unique_surveys:
        base_rates = get_base_rates(oof_ensemble, weights_train, survey_ids_train, [sid])
        pred_sum += base_rates[sid][key]
        actual_sum += actual_rates_by_survey[sid][key]
    if pred_sum > 0:
        final_corrections[k] = actual_sum / pred_sum

print("  Final corrections:")
for k in [0, 3, 7, 10, 14, 18]:
    pct = (k + 1) * 5
    adj = 1.0 + best_shrinkage * (final_corrections[k] - 1.0)
    print(f"    {pct}th: raw={final_corrections[k]:.4f}, shrunk={adj:.4f}")

# ================================================================
# PART 4: FINAL STRATEGY SELECTION
# ================================================================
print("\n" + "=" * 60)
print("PART 4: STRATEGY SELECTION")
print("=" * 60)

strategies = {
    f'soft_{best_base_name}': best_base_score,
    f'corrected (s={best_shrinkage})': best_corr_score,
}
# Also evaluate uncorrected alternatives if they're different
if best_base_name != 'global':
    strategies['soft_global'] = best_global_score
if best_base_name != 'per_t':
    strategies['soft_per_t'] = per_t_loso
if best_base_name != 'student':
    strategies['soft_student'] = best_student_score

for name, score in sorted(strategies.items(), key=lambda x: x[1]):
    marker = " <<<" if score == min(strategies.values()) else ""
    print(f"  {name:30s}: LOSO poverty wMAPE = {score:.4f}{marker}")

use_corrections = best_corr_score < best_base_score
if use_corrections:
    print(f"\n  Using corrected {best_base_name} with shrinkage={best_shrinkage}")
else:
    print(f"\n  Using uncorrected {best_base_name}")

# ================================================================
# PART 5: OOF VALIDATION
# ================================================================
print("\n" + "=" * 60)
print("PART 5: OOF VALIDATION")
print("=" * 60)

cons_mape_oof = np.mean(np.abs(oof_cons_best - y_train) / y_train) * 100

# Compute poverty rates
oof_pred_rates = get_base_rates(oof_ensemble, weights_train, survey_ids_train, unique_surveys)
if use_corrections:
    oof_pred_rates = apply_corrections(oof_pred_rates, final_corrections, best_shrinkage)

pov_oof = eval_poverty_only(oof_pred_rates, actual_rates_by_survey)
total_oof = 0.9 * pov_oof + 0.1 * cons_mape_oof

# LOSO-based estimate
loso_pov = best_corr_score if use_corrections else best_base_score
loso_total = 0.9 * loso_pov + 0.1 * cons_mape_oof

print(f"  Poverty rate wMAPE (OOF): {pov_oof:.4f}")
print(f"  Consumption MAPE (OOF):   {cons_mape_oof:.4f}")
print(f"  Total metric (OOF):       {total_oof:.4f}")
print(f"  LOSO poverty estimate:    {loso_pov:.4f}")
print(f"  LOSO-based total est.:    {loso_total:.4f}")
print(f"  #1 leaderboard:           3.207")

for sid in unique_surveys:
    print(f"\n  Survey {sid}:")
    for i, t in enumerate(THRESHOLDS):
        key = f'pct_hh_below_{t:.2f}'
        pct = (i + 1) * 5
        pred_r = oof_pred_rates[sid][key]
        actual_r = actual_rates_by_survey[sid][key]
        err = abs(pred_r - actual_r) / actual_r * 100 if actual_r > 0 else 0
        marker = " ***" if pct in [35, 40, 45] else ""
        if pct % 20 == 0 or pct in [5, 35, 40, 45]:
            print(f"    {pct:2d}th (${t:5.2f}): pred={pred_r:.4f} actual={actual_r:.4f} "
                  f"APE={err:6.2f}% (w={THRESHOLD_WEIGHTS[i]:.2f}){marker}")

# ================================================================
# PART 6: GENERATE TEST PREDICTIONS
# ================================================================
print("\n" + "=" * 60)
print("PART 6: TEST PREDICTIONS")
print("=" * 60)

test_surveys = [400000, 500000, 600000]

# Consumption: use MAPE-optimized weights
y_pred_cons = np.maximum(test_cons_best, 0.01)

# Poverty rates: use total-optimized weights + soft thresholds + corrections
test_pred_rates = get_base_rates(test_ensemble, test_weights, test_survey_ids, test_surveys)
if use_corrections:
    test_pred_rates = apply_corrections(test_pred_rates, final_corrections, best_shrinkage)

# ================================================================
# PART 7: SAVE
# ================================================================
print("\n" + "=" * 60)
print("PART 7: SAVING SUBMISSION")
print("=" * 60)

household_predictions = pd.DataFrame({
    'survey_id': test_encoded['survey_id'].values,
    'household_id': test_encoded['hhid'].values,
    'cons_ppp17': y_pred_cons
})

poverty_rates_list = []
for survey_id in test_surveys:
    rates = test_pred_rates[survey_id].copy()
    rates['survey_id'] = survey_id
    poverty_rates_list.append(rates)

    print(f"\n  Survey {survey_id}:")
    for i, t in enumerate(THRESHOLDS):
        key = f'pct_hh_below_{t:.2f}'
        pct = (i + 1) * 5
        if pct % 20 == 0 or pct in [5, 35, 40, 45]:
            print(f"    {pct:2d}th (${t:5.2f}): {rates[key]:.4f}")

poverty_df = pd.DataFrame(poverty_rates_list)
rate_cols = [f'pct_hh_below_{t:.2f}' for t in THRESHOLDS]
poverty_df = poverty_df[['survey_id'] + rate_cols]

household_predictions.to_csv(f"{DATA_DIR}/predicted_household_consumption.csv", index=False)
poverty_df.to_csv(f"{DATA_DIR}/predicted_poverty_distribution.csv", index=False)

import zipfile
with zipfile.ZipFile(f"{DATA_DIR}/submission.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(f"{DATA_DIR}/predicted_household_consumption.csv", "predicted_household_consumption.csv")
    zf.write(f"{DATA_DIR}/predicted_poverty_distribution.csv", "predicted_poverty_distribution.csv")

print("\n\nSubmission saved!")
print(f"\nEstimated score: {loso_total:.4f} (vs #1: 3.207)")
print("=" * 60)
print("DONE!")
print("=" * 60)
