#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS-inspired OHLC-only hybrid for next-day Close and direction forecasting.

Why this script exists
----------------------
The goal is to stay faithful to research integrity while still pushing directional
accuracy (DA) as high as possible using only OHLC data.

Design principles
-----------------
1. OHLC only:
   No volume, no news, no macro, no cross-asset features.

2. Strict chronology:
   Default split is 80/10/10 in time order.
   - 80% train
   - 10% validation
   - 10% test

3. No data leakage:
   - all feature transforms are causal
   - fuzzy regime model is fit on train only
   - global models are fit on train only
   - validation is used only for selecting the final blending / gating strategy
   - test stays untouched until the very end

4. ANFIS-inspired, but not literal classical ANFIS:
   Classical full-rule ANFIS can explode in rule count once the feature space grows.
   Here we keep the ANFIS spirit:
   - a fuzzy antecedent layer (soft regime memberships)
   - expert consequents conditioned on those memberships
   - a final weighted combination of expert outputs

5. Lightweight:
   Uses only numpy, pandas, scikit-learn.

Model sketch
------------
- Feature block: lagged OHLC returns + candle geometry + OHLC volatility/trend features
- Fuzzy regime block: GaussianMixture on a compact regime feature subset
- Global nonlinear models:
    * HistGradientBoostingRegressor for next-day Close log-return
    * HistGradientBoostingClassifier for next-day direction
- Regime experts:
    * weighted Ridge experts for return
    * weighted LogisticRegression experts for direction
- Final decision:
    * blend global + regime outputs
    * use classifier direction only when confidence is high enough
    * otherwise fall back to regression sign
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


OHLC = ("Open", "High", "Low", "Close")
EPSILON = 1e-9

MIN_TRAIN_ROWS = 400
MIN_VAL_ROWS = 60
MIN_TEST_ROWS = 60

ALPHA_GRID = np.array([0.25, 0.50, 0.75], dtype=np.float64)
CONFIDENCE_GRID = np.array([0.00, 0.10, 0.20, 0.30, 0.40], dtype=np.float64)
DIRECTION_THRESHOLD_GRID = np.array([0.45, 0.50, 0.55], dtype=np.float64)
MAGNITUDE_SHRINK_GRID = np.array([0.90, 1.00, 1.10], dtype=np.float64)

DEFAULT_REGIME_FEATURES = (
    "Close_ret1",
    "Close_ret5",
    "gap",
    "intraday",
    "range_pct",
    "gk_vol10",
    "ret_std10",
    "close_z10",
)


@dataclass(frozen=True)
class RunConfig:
    data_path: str
    stock: str
    output_dir: str
    min_date: str
    train_ratio: float
    val_ratio: float
    seed: int
    regime_grid: Tuple[int, ...]
    inner_cv_folds: int
    target_da: float
    target_r2: float
    disable_inner_cv: bool = False


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class HybridBundle:
    feature_cols: Tuple[str, ...]
    regime_cols: Tuple[str, ...]
    n_regimes: int
    regime_scaler: RobustScaler
    regime_model: GaussianMixture
    expert_scaler: RobustScaler
    global_regressor: HistGradientBoostingRegressor
    global_classifier: HistGradientBoostingClassifier
    reg_experts: List[Ridge]
    clf_experts: List[LogisticRegression]


def parse_grid(raw: str, cast_fn) -> Tuple:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast_fn(part))
    if not items:
        raise ValueError("Grid argument must contain at least one value.")
    return tuple(items)


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensure_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): ensure_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_jsonable(v) for v in obj]
    return to_python_scalar(obj)


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")


def prepare_ohlc_features(raw_df: pd.DataFrame, min_date: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    validate_input_columns(raw_df)

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    if min_date:
        start_date = pd.to_datetime(min_date, errors="coerce")
        if pd.notna(start_date):
            df = df[df["Date"] >= start_date].reset_index(drop=True)

    for col in OHLC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(OHLC)).reset_index(drop=True)

    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    close_ret1 = close.pct_change(1)
    gap = (open_ - close.shift(1)) / (close.shift(1) + EPSILON)
    intraday = (close - open_) / (open_ + EPSILON)
    range_pct = (high - low) / (close.shift(1) + EPSILON)

    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    log_hl = np.log((high + EPSILON) / (low + EPSILON))
    log_co = np.log((close + EPSILON) / (open_ + EPSILON))
    gk_var = 0.5 * np.square(log_hl) - (2.0 * np.log(2.0) - 1.0) * np.square(log_co)
    park_var = np.square(log_hl) / (4.0 * np.log(2.0))

    feature_dict: Dict[str, pd.Series] = {
        "gap": gap,
        "intraday": intraday,
        "range_pct": range_pct,
        "upper_wick": (high - body_top) / (close + EPSILON),
        "lower_wick": (body_bottom - low) / (close + EPSILON),
        "close_pos_in_range": (close - low) / ((high - low) + EPSILON),
        "open_pos_in_range": (open_ - low) / ((high - low) + EPSILON),
        "open_close_ratio": (open_ / (close + EPSILON)) - 1.0,
        "high_low_ratio": (high / (low + EPSILON)) - 1.0,
    }

    ohlc_series = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
    }

    for name, series in ohlc_series.items():
        for lag in (1, 2, 3, 5, 10):
            feature_dict[f"{name}_ret{lag}"] = series.pct_change(lag)
            feature_dict[f"{name}_logret{lag}"] = np.log((series + EPSILON) / (series.shift(lag) + EPSILON))
        for lag in (1, 2, 3, 5):
            feature_dict[f"{name}_lagratio{lag}"] = (series.shift(lag) / (close + EPSILON)) - 1.0

    for window in (3, 5, 10, 20):
        close_roll = close.rolling(window)
        feature_dict[f"close_z{window}"] = (close - close_roll.mean()) / (close_roll.std() + EPSILON)
        feature_dict[f"ret_mean{window}"] = close_ret1.rolling(window).mean()
        feature_dict[f"ret_std{window}"] = close_ret1.rolling(window).std()
        feature_dict[f"ret_skew{window}"] = close_ret1.rolling(window).skew()
        feature_dict[f"gk_vol{window}"] = np.sqrt(np.maximum(gk_var.rolling(window).mean(), 0.0))
        feature_dict[f"park_vol{window}"] = np.sqrt(np.maximum(park_var.rolling(window).mean(), 0.0))
        feature_dict[f"gap_mean{window}"] = gap.rolling(window).mean()
        feature_dict[f"body_mean{window}"] = intraday.rolling(window).mean()
        feature_dict[f"range_mean{window}"] = range_pct.rolling(window).mean()
        feature_dict[f"up_freq{window}"] = (close_ret1 > 0).rolling(window).mean()
        feature_dict[f"high_break{window}"] = (high.rolling(window).max() / (close + EPSILON)) - 1.0
        feature_dict[f"low_break{window}"] = (low.rolling(window).min() / (close + EPSILON)) - 1.0

    feature_df = pd.DataFrame(feature_dict)
    out = pd.concat([df[["Date"]], feature_df], axis=1)

    out["Close_current"] = close
    out["y_close"] = close.shift(-1)
    out["y_logret"] = np.log((close.shift(-1) + EPSILON) / (close + EPSILON))
    out["y_dir"] = (close.shift(-1) > close).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [
        col
        for col in out.columns
        if col not in {"Date", "Close_current", "y_close", "y_logret", "y_dir"}
    ]
    regime_cols = [col for col in DEFAULT_REGIME_FEATURES if col in out.columns]

    if not regime_cols:
        raise ValueError("Regime feature subset is empty after feature engineering.")

    return out, feature_cols, regime_cols


def build_time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> DatasetSplit:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    if train_end < MIN_TRAIN_ROWS:
        raise ValueError("Training split too small.")
    if (val_end - train_end) < MIN_VAL_ROWS:
        raise ValueError("Validation split too small.")
    if (n_rows - val_end) < MIN_TEST_ROWS:
        raise ValueError("Test split too small.")

    return DatasetSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


def build_expanding_folds(n_rows: int, n_folds: int) -> List[Tuple[slice, slice]]:
    if n_rows < (MIN_TRAIN_ROWS + MIN_VAL_ROWS):
        return []

    step = max(MIN_VAL_ROWS, (n_rows - MIN_TRAIN_ROWS) // (n_folds + 1))
    all_folds: List[Tuple[slice, slice]] = []
    train_end = MIN_TRAIN_ROWS

    while (train_end + MIN_VAL_ROWS) <= n_rows:
        val_end = min(n_rows, train_end + step)
        if (val_end - train_end) >= MIN_VAL_ROWS:
            all_folds.append((slice(0, train_end), slice(train_end, val_end)))
        train_end += step

    if len(all_folds) > n_folds:
        all_folds = all_folds[-n_folds:]

    return all_folds


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_close: np.ndarray,
    y_dir_pred: np.ndarray,
) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    non_zero_mask = np.abs(y_true) > EPSILON
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = 0.0

    y_dir_true = (y_true > current_close).astype(int)
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


def score_metrics(metrics: Dict[str, float]) -> float:
    return float(
        metrics["DA"]
        + (15.0 * metrics["R2"])
        - (0.15 * metrics["MAPE"])
        - (5.0 * abs(metrics["Precision"] - metrics["Recall"]))
    )


def compute_naive_baselines(history_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y_true = test_df["y_close"].to_numpy(dtype=np.float64)
    current_close = test_df["Close_current"].to_numpy(dtype=np.float64)

    # Baseline 1: tomorrow close equals today close.
    pred_persistence = current_close.copy()
    dir_persistence = np.zeros(len(test_df), dtype=int)
    persistence_metrics = compute_metrics(y_true, pred_persistence, current_close, dir_persistence)

    # Baseline 2: keep today's observed direction, use a train-only typical move size.
    if "Close_ret1" in test_df.columns:
        dir_trend = (test_df["Close_ret1"].to_numpy(dtype=np.float64) >= 0.0).astype(int)
    else:
        dir_trend = np.zeros(len(test_df), dtype=int)

    typical_abs_logret = float(np.nanmedian(np.abs(history_df["y_logret"].to_numpy(dtype=np.float64))))
    if not np.isfinite(typical_abs_logret) or typical_abs_logret < 1e-6:
        typical_abs_logret = 0.005

    pred_trend = current_close * np.exp(np.where(dir_trend == 1, typical_abs_logret, -typical_abs_logret))
    trend_metrics = compute_metrics(y_true, pred_trend, current_close, dir_trend)

    return {
        "close_persistence": persistence_metrics,
        "trend_persistence": trend_metrics,
    }


def fit_regime_model(train_df: pd.DataFrame, regime_cols: Sequence[str], n_regimes: int, seed: int) -> Tuple[RobustScaler, GaussianMixture]:
    scaler = RobustScaler()
    z_train = scaler.fit_transform(train_df.loc[:, list(regime_cols)].to_numpy(dtype=np.float64))

    try:
        regime_model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="diag",
            random_state=seed,
            n_init=3,
            reg_covar=1e-5,
        )
        regime_model.fit(z_train)
    except Exception:
        regime_model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="spherical",
            random_state=seed,
            n_init=5,
            reg_covar=1e-4,
        )
        regime_model.fit(z_train)

    return scaler, regime_model


def get_memberships(
    df_part: pd.DataFrame,
    regime_cols: Sequence[str],
    scaler: RobustScaler,
    regime_model: GaussianMixture,
) -> np.ndarray:
    z = scaler.transform(df_part.loc[:, list(regime_cols)].to_numpy(dtype=np.float64))
    weights = np.clip(regime_model.predict_proba(z), 1e-6, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float64)


def fit_hybrid_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    regime_cols: Sequence[str],
    n_regimes: int,
    seed: int,
) -> HybridBundle:
    x_train = train_df.loc[:, list(feature_cols)].to_numpy(dtype=np.float64)
    y_ret = train_df["y_logret"].to_numpy(dtype=np.float64)
    y_dir = train_df["y_dir"].to_numpy(dtype=int)

    regime_scaler, regime_model = fit_regime_model(train_df, regime_cols, n_regimes, seed)
    memberships = get_memberships(train_df, regime_cols, regime_scaler, regime_model)

    x_train_aug = np.hstack([x_train, memberships])

    global_regressor = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=180,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )
    global_regressor.fit(x_train_aug, y_ret)

    global_classifier = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=180,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )
    global_classifier.fit(x_train_aug, y_dir)

    expert_scaler = RobustScaler()
    x_train_scaled = expert_scaler.fit_transform(x_train)

    reg_experts: List[Ridge] = []
    clf_experts: List[LogisticRegression] = []

    for regime_id in range(n_regimes):
        sample_weight = np.clip(memberships[:, regime_id], 1e-6, 1.0)

        reg_expert = Ridge(alpha=3.0)
        reg_expert.fit(x_train_scaled, y_ret, sample_weight=sample_weight)
        reg_experts.append(reg_expert)

        clf_expert = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
        )
        try:
            clf_expert.fit(x_train_scaled, y_dir, sample_weight=sample_weight)
        except Exception:
            clf_expert = LogisticRegression(
                max_iter=400,
                solver="lbfgs",
            )
            clf_expert.fit(x_train_scaled, y_dir)
        clf_experts.append(clf_expert)

    return HybridBundle(
        feature_cols=tuple(feature_cols),
        regime_cols=tuple(regime_cols),
        n_regimes=int(n_regimes),
        regime_scaler=regime_scaler,
        regime_model=regime_model,
        expert_scaler=expert_scaler,
        global_regressor=global_regressor,
        global_classifier=global_classifier,
        reg_experts=reg_experts,
        clf_experts=clf_experts,
    )


def predict_components(bundle: HybridBundle, df_part: pd.DataFrame) -> Dict[str, np.ndarray]:
    x = df_part.loc[:, list(bundle.feature_cols)].to_numpy(dtype=np.float64)
    memberships = get_memberships(df_part, bundle.regime_cols, bundle.regime_scaler, bundle.regime_model)

    x_aug = np.hstack([x, memberships])
    global_ret = bundle.global_regressor.predict(x_aug)
    global_prob = bundle.global_classifier.predict_proba(x_aug)[:, 1]

    x_scaled = bundle.expert_scaler.transform(x)
    reg_matrix = np.column_stack([model.predict(x_scaled) for model in bundle.reg_experts])
    clf_matrix = np.column_stack([model.predict_proba(x_scaled)[:, 1] for model in bundle.clf_experts])

    regime_ret = np.sum(memberships * reg_matrix, axis=1)
    regime_prob = np.sum(memberships * clf_matrix, axis=1)

    return {
        "memberships": memberships,
        "global_ret": global_ret.astype(np.float64),
        "global_prob": global_prob.astype(np.float64),
        "regime_ret": regime_ret.astype(np.float64),
        "regime_prob": regime_prob.astype(np.float64),
    }


def blend_predictions(components: Dict[str, np.ndarray], alpha_reg: float, alpha_prob: float) -> Tuple[np.ndarray, np.ndarray]:
    blended_ret = (alpha_reg * components["regime_ret"]) + ((1.0 - alpha_reg) * components["global_ret"])
    blended_prob = (alpha_prob * components["regime_prob"]) + ((1.0 - alpha_prob) * components["global_prob"])
    return blended_ret, blended_prob


def apply_direction_strategy(
    current_close: np.ndarray,
    blended_ret: np.ndarray,
    blended_prob: np.ndarray,
    confidence_threshold: float,
    direction_threshold: float,
    magnitude_shrink: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjusted_ret = magnitude_shrink * blended_ret

    reg_direction = (adjusted_ret >= 0.0).astype(int)
    clf_direction = (blended_prob >= direction_threshold).astype(int)
    use_classifier = (np.abs(blended_prob - 0.5) * 2.0) >= confidence_threshold

    signed_ret = adjusted_ret.copy()
    signed_ret = np.where(
        use_classifier,
        np.where(clf_direction == 1, np.abs(adjusted_ret), -np.abs(adjusted_ret)),
        adjusted_ret,
    )
    final_direction = (signed_ret >= 0.0).astype(int)
    final_close = current_close * np.exp(signed_ret)

    return final_direction, final_close, signed_ret, use_classifier.astype(int)


def tune_strategy(valid_df: pd.DataFrame, components: Dict[str, np.ndarray]) -> Dict[str, Any]:
    current_close = valid_df["Close_current"].to_numpy(dtype=np.float64)
    y_true = valid_df["y_close"].to_numpy(dtype=np.float64)

    best_choice: Dict[str, Any] | None = None

    for alpha_reg in ALPHA_GRID:
        for alpha_prob in ALPHA_GRID:
            blended_ret, blended_prob = blend_predictions(components, float(alpha_reg), float(alpha_prob))

            for magnitude_shrink in MAGNITUDE_SHRINK_GRID:
                for confidence_threshold in CONFIDENCE_GRID:
                    for direction_threshold in DIRECTION_THRESHOLD_GRID:
                        final_dir, final_close, _, use_classifier = apply_direction_strategy(
                            current_close=current_close,
                            blended_ret=blended_ret,
                            blended_prob=blended_prob,
                            confidence_threshold=float(confidence_threshold),
                            direction_threshold=float(direction_threshold),
                            magnitude_shrink=float(magnitude_shrink),
                        )
                        metrics = compute_metrics(y_true, final_close, current_close, final_dir)
                        score = score_metrics(metrics)

                        candidate = {
                            "alpha_reg": float(alpha_reg),
                            "alpha_prob": float(alpha_prob),
                            "magnitude_shrink": float(magnitude_shrink),
                            "confidence_threshold": float(confidence_threshold),
                            "direction_threshold": float(direction_threshold),
                            "used_classifier_ratio": float(np.mean(use_classifier)),
                            "validation_score": float(score),
                            "validation_metrics": metrics,
                        }

                        if (best_choice is None) or (candidate["validation_score"] > best_choice["validation_score"]):
                            best_choice = candidate

    if best_choice is None:
        raise RuntimeError("Validation tuning failed to produce a valid strategy.")

    return best_choice


def evaluate_with_strategy(
    df_part: pd.DataFrame,
    components: Dict[str, np.ndarray],
    strategy: Dict[str, Any],
) -> Dict[str, Any]:
    current_close = df_part["Close_current"].to_numpy(dtype=np.float64)
    y_true = df_part["y_close"].to_numpy(dtype=np.float64)

    blended_ret, blended_prob = blend_predictions(
        components,
        alpha_reg=float(strategy["alpha_reg"]),
        alpha_prob=float(strategy["alpha_prob"]),
    )

    final_dir, final_close, signed_ret, use_classifier = apply_direction_strategy(
        current_close=current_close,
        blended_ret=blended_ret,
        blended_prob=blended_prob,
        confidence_threshold=float(strategy["confidence_threshold"]),
        direction_threshold=float(strategy["direction_threshold"]),
        magnitude_shrink=float(strategy["magnitude_shrink"]),
    )
    metrics = compute_metrics(y_true, final_close, current_close, final_dir)

    return {
        "metrics": metrics,
        "blended_ret": blended_ret,
        "blended_prob": blended_prob,
        "signed_ret": signed_ret,
        "final_dir": final_dir,
        "final_close": final_close,
        "use_classifier": use_classifier,
    }


def select_n_regimes_via_inner_cv(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    regime_cols: Sequence[str],
    regime_grid: Sequence[int],
    seed: int,
    n_folds: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    folds = build_expanding_folds(len(train_df), n_folds=n_folds)
    if not folds:
        fallback = int(regime_grid[0])
        return fallback, [
            {
                "n_regimes": fallback,
                "cv_score_mean": None,
                "cv_score_std": None,
                "note": "inner CV skipped because train window was too small",
            }
        ]

    summaries: List[Dict[str, Any]] = []
    best_n_regimes: int | None = None
    best_score: float | None = None

    for n_regimes in regime_grid:
        fold_scores: List[float] = []
        fold_metrics: List[Dict[str, float]] = []

        for fold_id, (train_slice, val_slice) in enumerate(folds):
            fold_train = train_df.iloc[train_slice].copy()
            fold_val = train_df.iloc[val_slice].copy()

            bundle = fit_hybrid_model(
                train_df=fold_train,
                feature_cols=feature_cols,
                regime_cols=regime_cols,
                n_regimes=int(n_regimes),
                seed=seed + fold_id,
            )
            components = predict_components(bundle, fold_val)

            blended_ret, blended_prob = blend_predictions(components, alpha_reg=0.50, alpha_prob=0.50)
            final_dir, final_close, _, _ = apply_direction_strategy(
                current_close=fold_val["Close_current"].to_numpy(dtype=np.float64),
                blended_ret=blended_ret,
                blended_prob=blended_prob,
                confidence_threshold=0.10,
                direction_threshold=0.50,
                magnitude_shrink=1.00,
            )
            metrics = compute_metrics(
                y_true=fold_val["y_close"].to_numpy(dtype=np.float64),
                y_pred=final_close,
                current_close=fold_val["Close_current"].to_numpy(dtype=np.float64),
                y_dir_pred=final_dir,
            )
            fold_scores.append(score_metrics(metrics))
            fold_metrics.append(metrics)

        score_mean = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        da_mean = float(np.mean([m["DA"] for m in fold_metrics]))
        r2_mean = float(np.mean([m["R2"] for m in fold_metrics]))
        mape_mean = float(np.mean([m["MAPE"] for m in fold_metrics]))

        summaries.append(
            {
                "n_regimes": int(n_regimes),
                "cv_score_mean": score_mean,
                "cv_score_std": score_std,
                "cv_da_mean": da_mean,
                "cv_r2_mean": r2_mean,
                "cv_mape_mean": mape_mean,
                "n_folds": int(len(fold_scores)),
            }
        )

        if (best_score is None) or (score_mean > best_score):
            best_score = score_mean
            best_n_regimes = int(n_regimes)

    if best_n_regimes is None:
        raise RuntimeError("Failed to select n_regimes via inner CV.")

    return best_n_regimes, summaries


def run_pipeline(raw_df: pd.DataFrame, cfg: RunConfig) -> Dict[str, Any]:
    full_df, feature_cols, regime_cols = prepare_ohlc_features(raw_df, min_date=cfg.min_date)
    split = build_time_split(full_df, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)

    if cfg.disable_inner_cv or len(cfg.regime_grid) == 1:
        selected_n_regimes = int(cfg.regime_grid[0])
        inner_cv_summary = [
            {
                "n_regimes": selected_n_regimes,
                "cv_score_mean": None,
                "cv_score_std": None,
                "note": "inner CV disabled by configuration",
            }
        ]
    else:
        selected_n_regimes, inner_cv_summary = select_n_regimes_via_inner_cv(
            train_df=split.train,
            feature_cols=feature_cols,
            regime_cols=regime_cols,
            regime_grid=cfg.regime_grid,
            seed=cfg.seed,
            n_folds=cfg.inner_cv_folds,
        )

    bundle = fit_hybrid_model(
        train_df=split.train,
        feature_cols=feature_cols,
        regime_cols=regime_cols,
        n_regimes=selected_n_regimes,
        seed=cfg.seed,
    )

    val_components = predict_components(bundle, split.val)
    best_strategy = tune_strategy(split.val, val_components)
    val_eval = evaluate_with_strategy(split.val, val_components, best_strategy)

    test_components = predict_components(bundle, split.test)
    test_eval = evaluate_with_strategy(split.test, test_components, best_strategy)

    baselines = compute_naive_baselines(
        history_df=pd.concat([split.train, split.val], axis=0),
        test_df=split.test,
    )

    prediction_df = pd.DataFrame(
        {
            "Date": split.test["Date"].dt.strftime("%Y-%m-%d"),
            "Close_current": split.test["Close_current"].to_numpy(dtype=np.float64),
            "y_true_close": split.test["y_close"].to_numpy(dtype=np.float64),
            "y_pred_close": test_eval["final_close"],
            "y_true_dir": split.test["y_dir"].to_numpy(dtype=int),
            "y_pred_dir": test_eval["final_dir"].astype(int),
            "global_prob_up": test_components["global_prob"],
            "regime_prob_up": test_components["regime_prob"],
            "blended_prob_up": test_eval["blended_prob"],
            "global_ret": test_components["global_ret"],
            "regime_ret": test_components["regime_ret"],
            "blended_ret": test_eval["blended_ret"],
            "final_signed_ret": test_eval["signed_ret"],
            "used_classifier_override": test_eval["use_classifier"].astype(int),
        }
    )

    memberships = test_components["memberships"]
    for regime_id in range(memberships.shape[1]):
        prediction_df[f"regime_w_{regime_id}"] = memberships[:, regime_id]

    target_reached_on_test = bool(
        (test_eval["metrics"]["DA"] >= cfg.target_da)
        and (test_eval["metrics"]["R2"] >= cfg.target_r2)
    )

    summary = {
        "stock": cfg.stock,
        "data_path": os.path.abspath(cfg.data_path),
        "config": ensure_jsonable(asdict(cfg)),
        "integrity_protocol": {
            "uses_only_ohlc": True,
            "chronological_split": True,
            "train_ratio": float(cfg.train_ratio),
            "val_ratio": float(cfg.val_ratio),
            "test_ratio": float(1.0 - cfg.train_ratio - cfg.val_ratio),
            "fit_on_train_only": True,
            "validation_used_for_selection_only": True,
            "test_untouched_until_final_evaluation": True,
            "inner_cv_on_train_only": bool(not cfg.disable_inner_cv and len(cfg.regime_grid) > 1),
            "split_counts": {
                "n_total": int(len(full_df)),
                "n_train": int(len(split.train)),
                "n_val": int(len(split.val)),
                "n_test": int(len(split.test)),
            },
        },
        "model_design": {
            "family": "ANFIS-inspired fuzzy regime mixture of experts",
            "feature_count": int(len(feature_cols)),
            "regime_feature_count": int(len(regime_cols)),
            "selected_n_regimes": int(selected_n_regimes),
            "regime_features": list(regime_cols),
            "global_models": {
                "regressor": "HistGradientBoostingRegressor",
                "classifier": "HistGradientBoostingClassifier",
            },
            "regime_experts": {
                "regressor": "weighted Ridge x n_regimes",
                "classifier": "weighted LogisticRegression x n_regimes",
            },
        },
        "inner_cv_summary": inner_cv_summary,
        "validation_selection": best_strategy,
        "validation_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "naive_baselines": baselines,
        "target": {
            "DA": float(cfg.target_da),
            "R2": float(cfg.target_r2),
        },
        "target_reached_on_test": target_reached_on_test,
    }

    return {
        "summary": ensure_jsonable(summary),
        "predictions": prediction_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ANFIS-inspired OHLC-only fuzzy-regime hybrid on one CSV."
    )
    parser.add_argument("--data-path", type=str, required=True, help="CSV path with Date/Open/High/Low/Close.")
    parser.add_argument("--stock", type=str, default="ASSET", help="Asset label for saved files.")
    parser.add_argument("--output-dir", type=str, default="anfis_fuzzy_regime_hybrid_outputs")
    parser.add_argument("--min-date", type=str, default="2015-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime-grid", type=str, default="3,4,5")
    parser.add_argument("--inner-cv-folds", type=int, default=3)
    parser.add_argument("--target-da", type=float, default=60.0)
    parser.add_argument("--target-r2", type=float, default=0.95)
    parser.add_argument(
        "--disable-inner-cv",
        action="store_true",
        help="Skip expanding-window CV inside the training block and use the first n_regimes value.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = RunConfig(
        data_path=args.data_path,
        stock=args.stock,
        output_dir=args.output_dir,
        min_date=args.min_date,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        regime_grid=tuple(parse_grid(args.regime_grid, int)),
        inner_cv_folds=int(args.inner_cv_folds),
        target_da=float(args.target_da),
        target_r2=float(args.target_r2),
        disable_inner_cv=bool(args.disable_inner_cv),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    raw_df = pd.read_csv(cfg.data_path)
    result = run_pipeline(raw_df, cfg)

    summary_path = os.path.join(cfg.output_dir, f"{cfg.stock}_anfis_fuzzy_regime_summary.json")
    prediction_path = os.path.join(cfg.output_dir, f"{cfg.stock}_anfis_fuzzy_regime_predictions.csv")

    with open(summary_path, "w", encoding="utf-8") as out_file:
        json.dump(result["summary"], out_file, indent=2)

    result["predictions"].to_csv(prediction_path, index=False)

    print("=" * 72)
    print("ANFIS-INSPIRED FUZZY-REGIME HYBRID (OHLC ONLY)")
    print("=" * 72)
    print(f"Data: {cfg.data_path}")
    print(f"Stock: {cfg.stock}")
    print(f"Split: train={cfg.train_ratio:.2f} | val={cfg.val_ratio:.2f} | test={1.0 - cfg.train_ratio - cfg.val_ratio:.2f}")
    print(f"Selected n_regimes: {result['summary']['model_design']['selected_n_regimes']}")
    print(
        "Validation -> "
        f"DA={result['summary']['validation_metrics']['DA']:.2f}% | "
        f"R2={result['summary']['validation_metrics']['R2']:.4f} | "
        f"MAPE={result['summary']['validation_metrics']['MAPE']:.3f}%"
    )
    print(
        "Test -> "
        f"DA={result['summary']['test_metrics']['DA']:.2f}% | "
        f"R2={result['summary']['test_metrics']['R2']:.4f} | "
        f"MAPE={result['summary']['test_metrics']['MAPE']:.3f}% | "
        f"target_reached={result['summary']['target_reached_on_test']}"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
