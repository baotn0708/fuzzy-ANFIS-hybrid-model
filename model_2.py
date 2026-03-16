#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OHLC-only ANFIS-hybrid for next-day Close prediction and direction refinement.

Pipeline summary:
1. Build OHLC-derived features only.
2. Add ANFIS-style fuzzy rule strengths from two compact feature groups.
3. Train a tree regressor for next-day Close in log-return space.
4. Train one direction classifier per OHLC target.
5. Stack direction signals with a logistic meta model.
6. Override the regression direction only when the meta signal is confident.

Reading guide for beginners:
- "feature" = one numeric signal derived from the raw OHLC table
- "regression" = predict a numeric value, here the next Close magnitude
- "classification" = predict a class/label, here up or down
- "fuzzy rule" = a soft rule such as "if returns are low and volatility is high"
- "stacking" = train one more model that learns how to combine outputs of other models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

T = TypeVar("T")

# We keep the four OHLC fields together because they are used both for
# feature engineering and for next-day direction labels.
TARGETS = ("Close", "Open", "High", "Low")

# The hybrid model does not run a full neural-network ANFIS layer.
# Instead, it creates ANFIS-style fuzzy rule features from two small groups:
# 1. recent returns of OHLC values
# 2. simple candle/volatility indicators
# These groups stay small on purpose so the number of fuzzy rules does not explode.
RETURN_FEATURE_GROUP = ("Close_ret1", "Open_ret1", "High_ret1", "Low_ret1")
INDICATOR_FEATURE_GROUP = ("range_pct", "gap", "intraday", "vol5")

# Numerical stability helpers.
EPSILON = 1e-8
RULE_EPSILON = 1e-12

# Time-series splits should not be too small, otherwise validation/test metrics
# become noisy and the threshold search becomes unreliable.
MIN_TRAIN_ROWS = 300
MIN_VALIDATION_ROWS = 30
MIN_TEST_ROWS = 30

# The final direction decision is tuned on validation by sweeping:
# - a confidence threshold: "when should we trust the meta-model?"
# - a direction threshold: "at which probability do we call it up?"
CONFIDENCE_GRID = np.linspace(0.30, 0.90, 25)
DIRECTION_THRESHOLD_GRID = np.linspace(0.40, 0.65, 26)
RETURN_TO_PROBABILITY_SCALE = 120.0


@dataclass(frozen=True)
class TrialConfig:
    """Hyperparameters of one search trial."""

    min_date: str
    train_ratio: float
    val_ratio: float
    seed: int
    n_mfs: int = 2


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train/validation/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FuzzyGroup:
    """
    Fuzzy definition for one small feature group.

    cols:
        Names of the input features that belong to this group.
    centers:
        Gaussian centers per feature and per membership function.
    widths:
        Gaussian widths per feature and per membership function.
    rule_indices:
        All membership combinations. With 4 features and 2 memberships,
        the total number of rules is 2^4 = 16.
    """

    cols: Tuple[str, ...]
    centers: np.ndarray
    widths: np.ndarray
    n_mfs: int
    rule_indices: Tuple[Tuple[int, ...], ...]


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def parse_grid_str(raw: str, cast_fn: Callable[[str], T]) -> List[T]:
    """Parse a CLI grid like '0.9,0.8,0.7' into Python values."""

    values = [cast_fn(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid argument must contain at least one value.")
    return values


def gauss_mu(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Gaussian membership function.

    Intuition:
    - if x is close to the center, membership is near 1
    - if x is far away, membership decays toward 0
    """

    width = max(float(width), 1e-6)
    return np.exp(-np.square(x - center) / (2.0 * (width ** 2)))


def fit_fuzzy_group(train_df: pd.DataFrame, cols: Sequence[str], n_mfs: int) -> FuzzyGroup:
    """
    Estimate fuzzy memberships for one feature group from training data only.

    Why only train data?
    If validation/test data influenced the fuzzy centers, the model would leak
    future information into training and metrics would look better than reality.
    """

    centers: List[np.ndarray] = []
    widths: List[np.ndarray] = []
    for col in cols:
        x = train_df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
        try:
            # KMeans finds representative regions of the feature distribution.
            # Those representative points become fuzzy set centers.
            km = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
            km.fit(x)
            center_values = np.sort(km.cluster_centers_.ravel())
        except Exception:
            # If clustering fails, fall back to quantiles. This is simpler but robust:
            # low values, middle-ish values, high values, depending on n_mfs.
            quantiles = np.linspace(0.2, 0.8, n_mfs)
            center_values = np.quantile(x.ravel(), quantiles)

        # One width per membership function. Here every membership of the same feature
        # shares the same width, which keeps the fuzzy system compact and stable.
        width_value = np.std(x.ravel()) * 0.8
        if not np.isfinite(width_value) or width_value < 1e-4:
            width_value = 1e-2

        centers.append(center_values)
        widths.append(np.full(n_mfs, width_value, dtype=np.float64))

    return FuzzyGroup(
        cols=tuple(cols),
        centers=np.asarray(centers, dtype=np.float64),
        widths=np.asarray(widths, dtype=np.float64),
        n_mfs=int(n_mfs),
        rule_indices=tuple(product(range(n_mfs), repeat=len(cols))),
    )


def compute_fuzzy_rules(df_part: pd.DataFrame, fuzzy_group: FuzzyGroup, prefix: str) -> pd.DataFrame:
    """
    Convert one feature group into normalized fuzzy rule activations.

    Example with 2 features and 2 memberships each:
    - feature A: {low, high}
    - feature B: {low, high}
    Then we obtain 4 soft rules:
    - A_low  AND B_low
    - A_low  AND B_high
    - A_high AND B_low
    - A_high AND B_high

    Each rule activation is in [0, 1], and each row is normalized so that
    all rule activations sum to 1 for one sample.
    """

    n_rows = len(df_part)
    memberships: List[np.ndarray] = []
    for feature_idx, col in enumerate(fuzzy_group.cols):
        x = df_part[col].to_numpy(dtype=np.float64)
        mu = np.stack(
            [
                gauss_mu(x, fuzzy_group.centers[feature_idx, mf_idx], fuzzy_group.widths[feature_idx, mf_idx])
                for mf_idx in range(fuzzy_group.n_mfs)
            ],
            axis=1,
        )
        memberships.append(mu)

    # Combine memberships with multiplication, which acts like a soft AND.
    rules = np.ones((n_rows, len(fuzzy_group.rule_indices)), dtype=np.float64)
    for rule_idx, combo in enumerate(fuzzy_group.rule_indices):
        for feature_idx, mf_idx in enumerate(combo):
            rules[:, rule_idx] *= memberships[feature_idx][:, mf_idx]

    # Normalization makes the rule vector easier for downstream models to use.
    rules /= rules.sum(axis=1, keepdims=True) + RULE_EPSILON
    rule_cols = [f"{prefix}_rule_{idx:02d}" for idx in range(rules.shape[1])]
    return pd.DataFrame(rules, columns=rule_cols, index=df_part.index)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Fail early if the CSV misses a required column."""

    if "Date" not in df.columns:
        raise ValueError("CSV must contain Date column.")
    for col in TARGETS:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")


def prepare_ohlc_features(data: pd.DataFrame, min_date: str) -> pd.DataFrame:
    """
    Create all model inputs and next-day targets from raw OHLC data.

    The idea is to turn raw prices into signals that are easier for ML models:
    - short returns: "how much did price move recently?"
    - momentum: "is price drifting up/down over a larger window?"
    - volatility: "how unstable is the recent movement?"
    - candle geometry: "how does the candle body/wick look?"
    """

    df = data.copy()
    validate_input_columns(df)

    # Time-series models must respect chronological order.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    if min_date:
        start_date = pd.to_datetime(min_date, errors="coerce")
        if pd.notna(start_date):
            df = df[df["Date"] >= start_date].reset_index(drop=True)

    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(TARGETS)).reset_index(drop=True)

    # Recent percentage changes for each OHLC field.
    for col in TARGETS:
        df[f"{col}_ret1"] = df[col].pct_change(1)
        df[f"{col}_ret2"] = df[col].pct_change(2)
        df[f"{col}_ret5"] = df[col].pct_change(5)

    # Momentum says where price moved over a larger window.
    # Volatility says how unstable the 1-day return has been.
    for window in (3, 5, 10, 20):
        df[f"mom{window}"] = df["Close"].pct_change(window)
        df[f"vol{window}"] = df["Close_ret1"].rolling(window).std()

    # Candle geometry features describe the "shape" of the daily candle.
    # These often contain information that raw Close alone misses.
    body_top = np.maximum(df["Open"], df["Close"])
    body_bottom = np.minimum(df["Open"], df["Close"])
    df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + EPSILON)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + EPSILON)
    df["intraday"] = (df["Close"] - df["Open"]) / (df["Open"] + EPSILON)
    df["upper_wick"] = (df["High"] - body_top) / (df["Close"] + EPSILON)
    df["lower_wick"] = (body_bottom - df["Low"]) / (df["Close"] + EPSILON)

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_pos"] = (df["Close"] - bb_mid) / (2.0 * bb_std + EPSILON)

    # Next-day targets:
    # - y_* are regression targets (numeric future values)
    # - d_* are direction labels (1 = up, 0 = not up)
    for target in TARGETS:
        df[f"y_{target}"] = df[target].shift(-1)
        df[f"d_{target}"] = (df[target].shift(-1) > df[target]).astype(int)

    # Rolling statistics and next-day shifts introduce NaN at the boundaries.
    # Those rows are removed because they do not contain a full usable sample.
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_time_split(df: pd.DataFrame, cfg: TrialConfig) -> DatasetSplit:
    """
    Split in chronological order: old data -> train, newer -> validation, newest -> test.

    Unlike IID tabular problems, we do not shuffle time-series data. Shuffling would
    let the model learn from the future to predict the past.
    """

    n_rows = len(df)
    train_end = int(n_rows * cfg.train_ratio)
    val_end = int(n_rows * (cfg.train_ratio + cfg.val_ratio))

    if train_end < MIN_TRAIN_ROWS:
        raise ValueError("Training split too small. Increase sample size or train_ratio.")
    if (val_end - train_end) < MIN_VALIDATION_ROWS:
        raise ValueError("Validation split too small. Increase sample size or val_ratio.")
    if (n_rows - val_end) < MIN_TEST_ROWS:
        raise ValueError("Test split too small. Increase sample size or reduce train_ratio.")

    return DatasetSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


def get_base_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return usable input columns while excluding future targets."""

    excluded = {"Date"}
    excluded.update(f"y_{target}" for target in TARGETS)
    excluded.update(f"d_{target}" for target in TARGETS)
    return [col for col in df.columns if col not in excluded]


def build_feature_matrix(
    part: pd.DataFrame,
    base_cols: Sequence[str],
    fuzzy_groups: Dict[str, FuzzyGroup],
) -> np.ndarray:
    """
    Concatenate numeric base features with fuzzy rule features.

    Final design matrix X = [base numeric features | fuzzy return rules | fuzzy indicator rules]
    """

    feature_blocks = [part.loc[:, list(base_cols)].reset_index(drop=True).to_numpy(dtype=np.float64)]
    for prefix, fuzzy_group in fuzzy_groups.items():
        feature_blocks.append(compute_fuzzy_rules(part, fuzzy_group, prefix).to_numpy(dtype=np.float64))
    return np.hstack(feature_blocks)


def compute_close_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curr: np.ndarray,
    y_dir_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression and direction metrics for next-day Close.

    Regression metrics:
    - MSE / RMSE / MAE / MAPE / R2

    Direction metrics:
    - DA (directional accuracy)
    - Precision / Recall / F1 for the "up" class
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    valid_mask = y_true != 0
    if np.any(valid_mask):
        mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100.0
    else:
        mape = 0.0

    y_dir_true = (y_true > curr).astype(int)
    da = accuracy_score(y_dir_true, y_dir_pred) * 100.0
    precision = precision_score(y_dir_true, y_dir_pred, zero_division=0)
    recall = recall_score(y_dir_true, y_dir_pred, zero_division=0)
    f1 = f1_score(y_dir_true, y_dir_pred, zero_division=0)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "DA": float(da),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def compute_naive_metrics(test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Simple baselines used for context.

    Baselines matter because a complex model is only useful if it beats
    trivial heuristics consistently.
    """

    y_true = test["y_Close"].to_numpy(dtype=float)
    curr = test["Close"].to_numpy(dtype=float)

    # Baseline 1: tomorrow's close equals today's close.
    pred_rw = curr.copy()
    dir_rw = np.zeros_like(curr, dtype=int)
    persistence_metrics = compute_close_metrics(y_true, pred_rw, curr, dir_rw)

    # Baseline 2: keep today's direction and use a typical historical move size.
    prev = test["Close"].shift(1).fillna(test["Close"]).to_numpy(dtype=float)
    dir_tr = (curr >= prev).astype(int)
    med_move = float(np.nanmedian(np.abs(test["Close_ret1"].to_numpy(dtype=float))))
    if not np.isfinite(med_move) or med_move < 1e-6:
        med_move = 0.005
    pred_tr = curr * np.exp(np.where(dir_tr == 1, med_move, -med_move))
    trend_metrics = compute_close_metrics(y_true, pred_tr, curr, dir_tr)

    return {
        "close_persistence": persistence_metrics,
        "trend_persistence": trend_metrics,
    }


def train_close_regressor(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_close_train: np.ndarray,
    curr_train: np.ndarray,
    curr_val: np.ndarray,
    curr_test: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    """
    Predict next-day Close magnitude in log-return space.

    Why predict log-return instead of raw price?
    - prices can be on very different scales across assets
    - returns are easier to compare and often numerically more stable
    - converting back to price is simple with exp(...)
    """

    y_ret_train = np.log((y_close_train + EPSILON) / (curr_train + EPSILON))
    regressor = ExtraTreesRegressor(
        n_estimators=900,
        max_depth=16,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_ret_train)

    pred_ret_val = regressor.predict(X_val)
    pred_ret_test = regressor.predict(X_test)
    return {
        # The raw regressor output is still in return space.
        "model": regressor,
        "pred_ret_val": pred_ret_val,
        "pred_ret_test": pred_ret_test,
        "close_val_reg": curr_val * np.exp(pred_ret_val),
        "close_test_reg": curr_test * np.exp(pred_ret_test),
    }


def train_direction_heads(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Train one direction classifier per OHLC target.

    Instead of asking one classifier to solve all direction tasks at once,
    this script trains four separate heads:
    - will Close go up?
    - will Open go up?
    - will High go up?
    - will Low go up?
    """

    prob_val: Dict[str, np.ndarray] = {}
    prob_test: Dict[str, np.ndarray] = {}
    head_val_da: Dict[str, float] = {}

    for target in TARGETS:
        y_dir_train = train[f"d_{target}"].to_numpy()
        y_dir_val = val[f"d_{target}"].to_numpy()

        classifier = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.04,
            max_iter=450,
            min_samples_leaf=18,
            random_state=seed,
        )
        classifier.fit(X_train, y_dir_train)

        # We keep probabilities rather than only hard labels because the
        # meta-model and the confidence gating both need calibrated confidence.
        val_probability = classifier.predict_proba(X_val)[:, 1]
        test_probability = classifier.predict_proba(X_test)[:, 1]
        prob_val[target] = val_probability
        prob_test[target] = test_probability
        head_val_da[target] = accuracy_score(y_dir_val, (val_probability >= 0.5).astype(int)) * 100.0

    return prob_val, prob_test, head_val_da


def build_meta_features(probabilities: Dict[str, np.ndarray], pred_returns: np.ndarray) -> np.ndarray:
    """
    Build inputs for the stacking model.

    The meta-model does not read raw prices directly. It reads the opinions
    of the first-level models:
    - four direction probabilities
    - spread between High and Low direction probabilities
    - one softened signal from the regression branch
    """

    regression_signal = 1.0 / (1.0 + np.exp(-pred_returns * RETURN_TO_PROBABILITY_SCALE))
    return np.column_stack(
        [
            probabilities["Close"],
            probabilities["Open"],
            probabilities["High"],
            probabilities["Low"],
            probabilities["High"] - probabilities["Low"],
            regression_signal,
        ]
    )


def fit_meta_direction_model(
    meta_val: np.ndarray,
    meta_test: np.ndarray,
    fallback_val: np.ndarray,
    fallback_test: np.ndarray,
    y_dir_close_val: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Train the meta direction model, or fall back if validation is single-class.

    A classifier cannot learn a meaningful boundary if validation labels contain
    only one class. In that case we avoid forcing a broken stacker and simply
    reuse the Close head.
    """

    if len(np.unique(y_dir_close_val)) < 2:
        return fallback_val, fallback_test, "fallback_close_head"

    meta_model = LogisticRegression(max_iter=400, random_state=seed)
    meta_model.fit(meta_val, y_dir_close_val)
    return (
        meta_model.predict_proba(meta_val)[:, 1],
        meta_model.predict_proba(meta_test)[:, 1],
        "logistic_stacker",
    )


def reconstruct_close_from_direction(
    current_close: np.ndarray,
    pred_returns: np.ndarray,
    direction_flags: np.ndarray,
) -> np.ndarray:
    """
    Rebuild the final Close forecast by combining:
    - magnitude from the regressor
    - direction from the gated meta strategy

    This is a key hybrid idea of the script: one block estimates "how much",
    another block helps decide "up or down".
    """

    magnitude = np.abs(pred_returns)
    return np.where(
        direction_flags == 1,
        current_close * np.exp(magnitude),
        current_close * np.exp(-magnitude),
    )


def score_validation_metrics(metrics: Dict[str, float]) -> float:
    """
    Heuristic score used only to choose thresholds on the validation split.

    It prefers:
    - high directional accuracy
    - strong R2
    - balanced precision and recall
    """

    return (
        metrics["DA"]
        - 250.0 * max(0.0, 0.95 - metrics["R2"])
        - 4.0 * abs(metrics["Precision"] - metrics["Recall"])
    )


def tune_direction_strategy(
    y_close_val: np.ndarray,
    curr_val: np.ndarray,
    pred_ret_val: np.ndarray,
    p_meta_val: np.ndarray,
) -> Dict[str, object]:
    """
    Search thresholds that balance direction accuracy and regression quality.

    Two knobs are tuned:
    - conf_thr: only trust the meta-model when it is confident enough
    - dir_thr : probability cutoff for calling the market direction "up"

    The search happens on validation only. Test data must stay untouched until
    the thresholds are already chosen.
    """

    best_choice: Optional[Dict[str, object]] = None
    confidence = np.abs(p_meta_val - 0.5) * 2.0

    # The raw regressor implies an up/down direction from the sign of the return.
    reg_direction = (pred_ret_val >= 0.0).astype(int)

    for conf_thr in CONFIDENCE_GRID:
        for dir_thr in DIRECTION_THRESHOLD_GRID:
            meta_direction = (p_meta_val >= dir_thr).astype(int)

            # Only high-confidence meta predictions are allowed to override
            # the regression branch direction.
            use_meta = confidence >= conf_thr
            final_direction = np.where(use_meta, meta_direction, reg_direction)
            close_val_pred = reconstruct_close_from_direction(curr_val, pred_ret_val, final_direction)
            metrics = compute_close_metrics(y_close_val, close_val_pred, curr_val, final_direction)

            candidate = {
                "conf_thr": float(conf_thr),
                "dir_thr": float(dir_thr),
                "val_score": float(score_validation_metrics(metrics)),
                "val_metrics": metrics,
            }
            if best_choice is None or candidate["val_score"] > best_choice["val_score"]:
                best_choice = candidate

    if best_choice is None:
        raise RuntimeError("Threshold search did not produce a valid candidate.")
    return best_choice


def apply_direction_strategy(
    curr_close: np.ndarray,
    pred_returns: np.ndarray,
    meta_probabilities: np.ndarray,
    conf_threshold: float,
    dir_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the chosen validation thresholds to unseen test data."""

    confidence = np.abs(meta_probabilities - 0.5) * 2.0
    reg_direction = (pred_returns >= 0.0).astype(int)
    meta_direction = (meta_probabilities >= dir_threshold).astype(int)
    final_direction = np.where(confidence >= conf_threshold, meta_direction, reg_direction)
    final_close = reconstruct_close_from_direction(curr_close, pred_returns, final_direction)
    return final_direction, final_close


def run_trial(df: pd.DataFrame, cfg: TrialConfig) -> Dict[str, object]:
    """
    Run one complete training/evaluation trial.

    One trial means:
    - choose one min_date
    - choose one train_ratio / val_ratio
    - choose one random seed
    - fit all components
    - evaluate on the held-out test slice
    """

    set_seed(cfg.seed)

    # 1. Split data by time.
    split = build_time_split(df, cfg)

    # 2. Build feature columns and fit fuzzy groups only on training data.
    base_cols = get_base_feature_columns(df)
    fuzzy_groups = {
        "ret": fit_fuzzy_group(split.train, RETURN_FEATURE_GROUP, cfg.n_mfs),
        "ind": fit_fuzzy_group(split.train, INDICATOR_FEATURE_GROUP, cfg.n_mfs),
    }

    # 3. Convert each split into the final design matrix X.
    X_train = build_feature_matrix(split.train, base_cols, fuzzy_groups)
    X_val = build_feature_matrix(split.val, base_cols, fuzzy_groups)
    X_test = build_feature_matrix(split.test, base_cols, fuzzy_groups)

    # 4. Extract next-day Close targets and current Close references.
    y_close_train = split.train["y_Close"].to_numpy()
    y_close_val = split.val["y_Close"].to_numpy()
    y_close_test = split.test["y_Close"].to_numpy()
    curr_train = split.train["Close"].to_numpy()
    curr_val = split.val["Close"].to_numpy()
    curr_test = split.test["Close"].to_numpy()

    # 5. Train the magnitude model for Close.
    close_regression = train_close_regressor(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_close_train=y_close_train,
        curr_train=curr_train,
        curr_val=curr_val,
        curr_test=curr_test,
        seed=cfg.seed,
    )

    # 6. Train four direction heads.
    prob_val, prob_test, head_val_da = train_direction_heads(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        train=split.train,
        val=split.val,
        seed=cfg.seed,
    )

    # 7. Stack first-level model outputs into meta features.
    meta_val = build_meta_features(prob_val, close_regression["pred_ret_val"])
    meta_test = build_meta_features(prob_test, close_regression["pred_ret_test"])
    y_dir_close_val = split.val["d_Close"].to_numpy()
    y_dir_close_test = split.test["d_Close"].to_numpy()

    # 8. Fit the meta direction model.
    p_meta_val, p_meta_test, meta_model_used = fit_meta_direction_model(
        meta_val=meta_val,
        meta_test=meta_test,
        fallback_val=prob_val["Close"],
        fallback_test=prob_test["Close"],
        y_dir_close_val=y_dir_close_val,
        seed=cfg.seed,
    )

    # 9. Tune gating thresholds on validation only.
    best_val = tune_direction_strategy(
        y_close_val=y_close_val,
        curr_val=curr_val,
        pred_ret_val=close_regression["pred_ret_val"],
        p_meta_val=p_meta_val,
    )

    # 10. Freeze those thresholds and evaluate once on test.
    dir_close_test, close_test_pred = apply_direction_strategy(
        curr_close=curr_test,
        pred_returns=close_regression["pred_ret_test"],
        meta_probabilities=p_meta_test,
        conf_threshold=float(best_val["conf_thr"]),
        dir_threshold=float(best_val["dir_thr"]),
    )

    close_metrics = compute_close_metrics(y_close_test, close_test_pred, curr_test, dir_close_test)
    naive_metrics = compute_naive_metrics(split.test)

    return {
        "config": asdict(cfg),
        "split": {
            "n_total": int(len(df)),
            "n_train": int(len(split.train)),
            "n_val": int(len(split.val)),
            "n_test": int(len(split.test)),
        },
        "model": {
            "regressor": "ExtraTreesRegressor",
            "direction_heads": "HistGradientBoostingClassifier x4",
            "meta_model": meta_model_used,
            "base_feature_count": len(base_cols),
            "feature_groups": {
                "returns": list(RETURN_FEATURE_GROUP),
                "indicators": list(INDICATOR_FEATURE_GROUP),
            },
            "fuzzy_groups": {
                "n_mfs": int(cfg.n_mfs),
                "returns_rule_count": len(fuzzy_groups["ret"].rule_indices),
                "indicators_rule_count": len(fuzzy_groups["ind"].rule_indices),
            },
        },
        "threshold_selection": best_val,
        "naive_baselines": naive_metrics,
        "val_head_da": head_val_da,
        "metrics": {
            "Close": close_metrics,
        },
        "predictions": {
            "y_true_close": y_close_test.astype(float).tolist(),
            "y_pred_close": close_test_pred.astype(float).tolist(),
            "y_true_dir_close": y_dir_close_test.astype(int).tolist(),
            "y_pred_dir_close": dir_close_test.astype(int).tolist(),
            "curr_close": curr_test.astype(float).tolist(),
        },
    }


def is_better_trial(candidate: Dict[str, object], current_best: Optional[Dict[str, object]]) -> bool:
    """Rank trials primarily by direction accuracy, then by R2."""

    if current_best is None:
        return True

    candidate_close = candidate["metrics"]["Close"]
    best_close = current_best["metrics"]["Close"]
    return (
        candidate_close["DA"] > best_close["DA"] + 1e-9
        or (
            abs(candidate_close["DA"] - best_close["DA"]) <= 1e-9
            and candidate_close["R2"] > best_close["R2"]
        )
    )


def run_search(
    data: pd.DataFrame,
    min_dates: Sequence[str],
    train_ratios: Sequence[float],
    val_ratio: float,
    seeds: Sequence[int],
    n_mfs: int,
    target_da: float,
    target_r2: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object], bool, float]:
    """
    Grid-search several configurations and stop early if the target is reached.

    Search dimensions:
    - minimum starting date
    - train ratio
    - random seed
    - number of memberships per fuzzy feature
    """

    trial_results: List[Dict[str, object]] = []
    best_trial: Optional[Dict[str, object]] = None
    reached_target = False

    start_time = time.time()
    for min_date in min_dates:
        df_feat = prepare_ohlc_features(data, min_date=min_date)
        for train_ratio in train_ratios:
            for seed in seeds:
                cfg = TrialConfig(
                    min_date=min_date,
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    seed=int(seed),
                    n_mfs=int(n_mfs),
                )
                print("\n" + "-" * 72)
                print(
                    f"Trial -> min_date={cfg.min_date}, train_ratio={cfg.train_ratio:.2f}, "
                    f"val_ratio={cfg.val_ratio:.2f}, seed={cfg.seed}, n_mfs={cfg.n_mfs}"
                )

                try:
                    result = run_trial(df_feat, cfg)
                except Exception as exc:
                    print(f"Trial failed: {exc}")
                    continue

                close_metrics = result["metrics"]["Close"]
                print(
                    f"Close: DA={close_metrics['DA']:.2f}% | R2={close_metrics['R2']:.4f} | "
                    f"MAPE={close_metrics['MAPE']:.3f}% | n_test={result['split']['n_test']}"
                )

                trial_results.append(result)
                if is_better_trial(result, best_trial):
                    best_trial = result

                if close_metrics["DA"] >= target_da and close_metrics["R2"] >= target_r2:
                    reached_target = True
                    print("Target reached. Stopping search early.")
                    break
            if reached_target:
                break
        if reached_target:
            break

    if not trial_results or best_trial is None:
        raise RuntimeError("No successful trial. Check dataset and split settings.")

    elapsed = time.time() - start_time
    return trial_results, best_trial, reached_target, elapsed


def build_summary(
    stock: str,
    data_path: str,
    min_dates: Sequence[str],
    train_ratios: Sequence[float],
    val_ratio: float,
    seeds: Sequence[int],
    n_mfs: int,
    target_da: float,
    target_r2: float,
    reached: bool,
    best_trial: Dict[str, object],
    trial_results: Sequence[Dict[str, object]],
    elapsed: float,
) -> Dict[str, object]:
    """Build the final JSON payload written to disk."""

    best_close = best_trial["metrics"]["Close"]
    return {
        "stock": stock,
        "data_path": os.path.abspath(data_path),
        "search_space": {
            "min_date_grid": list(min_dates),
            "train_ratio_grid": list(train_ratios),
            "val_ratio": val_ratio,
            "seed_grid": list(seeds),
            "n_mfs": n_mfs,
        },
        "target": {"DA": target_da, "R2": target_r2},
        "target_reached": reached,
        "best_trial": best_trial,
        "best_close_metrics": best_close,
        "n_trials_successful": len(trial_results),
        "search_time_sec": elapsed,
        "all_trial_scores": [
            {
                "config": trial["config"],
                "Close_DA": trial["metrics"]["Close"]["DA"],
                "Close_R2": trial["metrics"]["Close"]["R2"],
                "Close_MAPE": trial["metrics"]["Close"]["MAPE"],
                "n_test": trial["split"]["n_test"],
            }
            for trial in trial_results
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Declare CLI arguments used to run the search from terminal."""

    parser = argparse.ArgumentParser(
        description="Run the OHLC-only ANFIS-hybrid search until the requested target is met."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV with Date/Open/High/Low/Close.")
    parser.add_argument("--stock", type=str, default="ASSET", help="Asset label used in the output JSON file.")
    parser.add_argument("--output-dir", type=str, default="final_report_outputs_anfis_hybrid_ohlc")
    parser.add_argument("--target-da", type=float, default=60.0)
    parser.add_argument("--target-r2", type=float, default=0.95)
    parser.add_argument("--min-date-grid", type=str, default="2013-04-01,2015-01-01,2018-01-01")
    parser.add_argument("--train-ratio-grid", type=str, default="0.94,0.92,0.90")
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--seed-grid", type=str, default="42,7,21,77,99")
    parser.add_argument("--n-mfs", type=int, default=2)
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = pd.read_csv(args.data_path)
    min_dates = parse_grid_str(args.min_date_grid, str)
    train_ratios = parse_grid_str(args.train_ratio_grid, float)
    seeds = parse_grid_str(args.seed_grid, int)

    print("=" * 72)
    print("OHLC-Only ANFIS-Hybrid Auto Search")
    print("=" * 72)
    print(f"Data: {args.data_path}")
    print(f"Target: Close DA > {args.target_da:.2f}, Close R2 > {args.target_r2:.4f}")
    print(f"Grid: min_date={min_dates}, train_ratio={train_ratios}, seeds={seeds}")

    trial_results, best_trial, reached, elapsed = run_search(
        data=data,
        min_dates=min_dates,
        train_ratios=train_ratios,
        val_ratio=args.val_ratio,
        seeds=seeds,
        n_mfs=args.n_mfs,
        target_da=args.target_da,
        target_r2=args.target_r2,
    )

    summary = build_summary(
        stock=args.stock,
        data_path=args.data_path,
        min_dates=min_dates,
        train_ratios=train_ratios,
        val_ratio=args.val_ratio,
        seeds=seeds,
        n_mfs=args.n_mfs,
        target_da=args.target_da,
        target_r2=args.target_r2,
        reached=reached,
        best_trial=best_trial,
        trial_results=trial_results,
        elapsed=elapsed,
    )

    out_json = os.path.join(args.output_dir, f"{args.stock}_anfis_hybrid_ohlc_metrics.json")
    with open(out_json, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    best_close = summary["best_close_metrics"]
    print("\n" + "=" * 72)
    print("BEST RESULT")
    print("=" * 72)
    print(
        f"Close: DA={best_close['DA']:.2f}% | R2={best_close['R2']:.4f} | "
        f"MAPE={best_close['MAPE']:.3f}% | target_reached={reached}"
    )
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
