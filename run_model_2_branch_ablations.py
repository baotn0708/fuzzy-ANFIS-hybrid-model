#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled ablations for model_2.py.

We compare three variants on the same chronological split and config:
1. full           : regression branch + direction branch
2. price_only     : regression branch only
3. direction_only : direction branch only, with a train-only typical move size

This keeps the comparison focused on whether one branch is helping or hurting.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import model_2 as base


def estimate_typical_abs_return(y_close_train: np.ndarray, curr_train: np.ndarray) -> float:
    """Estimate a train-only move size for the direction-only ablation."""

    log_returns = np.log((y_close_train + base.EPSILON) / (curr_train + base.EPSILON))
    typical_abs_return = float(np.nanmedian(np.abs(log_returns)))
    if not np.isfinite(typical_abs_return) or typical_abs_return < 1e-6:
        typical_abs_return = 0.005
    return typical_abs_return


def reconstruct_close_from_fixed_return(
    current_close: np.ndarray,
    direction_flags: np.ndarray,
    abs_return: float,
) -> np.ndarray:
    """Convert a direction-only signal into a price using a fixed train-only move size."""

    signed_return = np.where(direction_flags == 1, abs_return, -abs_return)
    return current_close * np.exp(signed_return)


def build_direction_only_meta_features(probabilities: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack only direction opinions, without the regression branch signal."""

    return np.column_stack(
        [
            probabilities["Close"],
            probabilities["Open"],
            probabilities["High"],
            probabilities["Low"],
            probabilities["High"] - probabilities["Low"],
        ]
    )


def tune_direction_only_strategy(
    y_close_val: np.ndarray,
    curr_val: np.ndarray,
    fallback_direction: np.ndarray,
    meta_probabilities: np.ndarray,
    abs_return: float,
) -> Dict[str, object]:
    """Tune the direction-only branch on validation without using the regressor."""

    best_choice = None
    confidence = np.abs(meta_probabilities - 0.5) * 2.0

    for conf_thr in base.CONFIDENCE_GRID:
        for dir_thr in base.DIRECTION_THRESHOLD_GRID:
            meta_direction = (meta_probabilities >= dir_thr).astype(int)
            final_direction = np.where(confidence >= conf_thr, meta_direction, fallback_direction)
            close_val_pred = reconstruct_close_from_fixed_return(curr_val, final_direction, abs_return)
            metrics = base.compute_close_metrics(y_close_val, close_val_pred, curr_val, final_direction)
            candidate = {
                "conf_thr": float(conf_thr),
                "dir_thr": float(dir_thr),
                "val_score": float(base.score_validation_metrics(metrics)),
                "val_metrics": metrics,
            }
            if best_choice is None or candidate["val_score"] > best_choice["val_score"]:
                best_choice = candidate

    if best_choice is None:
        raise RuntimeError("Direction-only threshold search did not produce a valid candidate.")
    return best_choice


def apply_direction_only_strategy(
    curr_close: np.ndarray,
    fallback_direction: np.ndarray,
    meta_probabilities: np.ndarray,
    abs_return: float,
    conf_threshold: float,
    dir_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the direction-only ablation with train-only fixed move size."""

    confidence = np.abs(meta_probabilities - 0.5) * 2.0
    meta_direction = (meta_probabilities >= dir_threshold).astype(int)
    final_direction = np.where(confidence >= conf_threshold, meta_direction, fallback_direction)
    final_close = reconstruct_close_from_fixed_return(curr_close, final_direction, abs_return)
    return final_direction, final_close


def run_branch_ablation(df: pd.DataFrame, cfg: base.TrialConfig) -> Dict[str, object]:
    """Run one controlled ablation suite on a fixed split/config."""

    base.set_seed(cfg.seed)
    split = base.build_time_split(df, cfg)
    meta_train_slice, threshold_tune_slice = base.build_meta_validation_slices(len(split.val))

    base_cols = base.get_base_feature_columns(df)
    fuzzy_groups = {
        "ret": base.fit_fuzzy_group(split.train, base.RETURN_FEATURE_GROUP, cfg.n_mfs),
        "ind": base.fit_fuzzy_group(split.train, base.INDICATOR_FEATURE_GROUP, cfg.n_mfs),
    }

    X_train = base.build_feature_matrix(split.train, base_cols, fuzzy_groups)
    X_val = base.build_feature_matrix(split.val, base_cols, fuzzy_groups)
    X_test = base.build_feature_matrix(split.test, base_cols, fuzzy_groups)

    y_close_train = split.train["y_Close"].to_numpy()
    y_close_val = split.val["y_Close"].to_numpy()
    y_close_test = split.test["y_Close"].to_numpy()
    curr_train = split.train["Close"].to_numpy()
    curr_val = split.val["Close"].to_numpy()
    curr_test = split.test["Close"].to_numpy()

    close_regression = base.train_close_regressor(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_close_train=y_close_train,
        curr_train=curr_train,
        curr_val=curr_val,
        curr_test=curr_test,
        seed=cfg.seed,
    )

    prob_val, prob_test, head_val_da = base.train_direction_heads(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        train=split.train,
        val=split.val,
        seed=cfg.seed,
    )
    if prob_test is None or close_regression["pred_ret_test"] is None or close_regression["close_test_reg"] is None:
        raise RuntimeError("Expected test predictions for the ablation run.")

    y_dir_close_val = split.val["d_Close"].to_numpy()
    prob_val_meta_train = base.slice_probability_dict(prob_val, meta_train_slice)
    prob_val_tune = base.slice_probability_dict(prob_val, threshold_tune_slice)
    y_dir_close_meta_train = y_dir_close_val[meta_train_slice]
    y_close_tune = y_close_val[threshold_tune_slice]
    curr_tune = curr_val[threshold_tune_slice]

    typical_abs_return = estimate_typical_abs_return(y_close_train, curr_train)
    naive_metrics = base.compute_naive_metrics(pd.concat([split.train, split.val], axis=0), split.test)

    full_meta_train = base.build_meta_features(
        prob_val_meta_train,
        close_regression["pred_ret_val"][meta_train_slice],
    )
    full_meta_tune = base.build_meta_features(
        prob_val_tune,
        close_regression["pred_ret_val"][threshold_tune_slice],
    )
    full_meta_test = base.build_meta_features(prob_test, close_regression["pred_ret_test"])
    p_full_tune, p_full_test, full_meta_model = base.fit_meta_direction_model(
        meta_train=full_meta_train,
        meta_tune=full_meta_tune,
        meta_test=full_meta_test,
        fallback_tune=prob_val["Close"][threshold_tune_slice],
        fallback_test=prob_test["Close"],
        y_dir_close_meta_train=y_dir_close_meta_train,
        seed=cfg.seed,
    )
    full_selection = base.tune_direction_strategy(
        y_close_val=y_close_tune,
        curr_val=curr_tune,
        pred_ret_val=close_regression["pred_ret_val"][threshold_tune_slice],
        p_meta_val=p_full_tune,
    )
    full_test_direction, full_test_close = base.apply_direction_strategy(
        curr_close=curr_test,
        pred_returns=close_regression["pred_ret_test"],
        meta_probabilities=p_full_test,
        conf_threshold=float(full_selection["conf_thr"]),
        dir_threshold=float(full_selection["dir_thr"]),
    )
    full_test_metrics = base.compute_close_metrics(y_close_test, full_test_close, curr_test, full_test_direction)

    price_only_direction_tune = (close_regression["pred_ret_val"][threshold_tune_slice] >= 0.0).astype(int)
    price_only_selection_metrics = base.compute_close_metrics(
        y_close_tune,
        close_regression["close_val_reg"][threshold_tune_slice],
        curr_tune,
        price_only_direction_tune,
    )
    price_only_test_direction = (close_regression["pred_ret_test"] >= 0.0).astype(int)
    price_only_test_metrics = base.compute_close_metrics(
        y_close_test,
        close_regression["close_test_reg"],
        curr_test,
        price_only_test_direction,
    )

    direction_only_meta_train = build_direction_only_meta_features(prob_val_meta_train)
    direction_only_meta_tune = build_direction_only_meta_features(prob_val_tune)
    direction_only_meta_test = build_direction_only_meta_features(prob_test)
    p_direction_only_tune, p_direction_only_test, direction_only_meta_model = base.fit_meta_direction_model(
        meta_train=direction_only_meta_train,
        meta_tune=direction_only_meta_tune,
        meta_test=direction_only_meta_test,
        fallback_tune=prob_val["Close"][threshold_tune_slice],
        fallback_test=prob_test["Close"],
        y_dir_close_meta_train=y_dir_close_meta_train,
        seed=cfg.seed,
    )
    direction_only_fallback_tune = (prob_val["Close"][threshold_tune_slice] >= 0.5).astype(int)
    direction_only_selection = tune_direction_only_strategy(
        y_close_val=y_close_tune,
        curr_val=curr_tune,
        fallback_direction=direction_only_fallback_tune,
        meta_probabilities=p_direction_only_tune,
        abs_return=typical_abs_return,
    )
    direction_only_fallback_test = (prob_test["Close"] >= 0.5).astype(int)
    direction_only_test_direction, direction_only_test_close = apply_direction_only_strategy(
        curr_close=curr_test,
        fallback_direction=direction_only_fallback_test,
        meta_probabilities=p_direction_only_test,
        abs_return=typical_abs_return,
        conf_threshold=float(direction_only_selection["conf_thr"]),
        dir_threshold=float(direction_only_selection["dir_thr"]),
    )
    direction_only_test_metrics = base.compute_close_metrics(
        y_close_test,
        direction_only_test_close,
        curr_test,
        direction_only_test_direction,
    )

    return {
        "config": {
            "min_date": cfg.min_date,
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "n_mfs": cfg.n_mfs,
        },
        "split": {
            "n_total": int(len(df)),
            "n_train": int(len(split.train)),
            "n_val": int(len(split.val)),
            "n_val_meta_train": int(meta_train_slice.stop - meta_train_slice.start),
            "n_val_tune": int(threshold_tune_slice.stop - threshold_tune_slice.start),
            "n_test": int(len(split.test)),
        },
        "train_statistics": {
            "typical_abs_close_log_return": typical_abs_return,
        },
        "val_head_da": head_val_da,
        "naive_baselines": naive_metrics,
        "modes": {
            "full": {
                "meta_model": full_meta_model,
                "threshold_selection": full_selection,
                "selection_metrics": full_selection["val_metrics"],
                "test_metrics": full_test_metrics,
            },
            "price_only": {
                "selection_metrics": price_only_selection_metrics,
                "test_metrics": price_only_test_metrics,
            },
            "direction_only": {
                "meta_model": direction_only_meta_model,
                "threshold_selection": direction_only_selection,
                "selection_metrics": direction_only_selection["val_metrics"],
                "test_metrics": direction_only_test_metrics,
            },
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled branch ablations for model_2.py.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV with Date/Open/High/Low/Close.")
    parser.add_argument("--stock", type=str, default="ASSET", help="Asset label for the output JSON file.")
    parser.add_argument("--output-dir", type=str, default="model_2_branch_ablation_outputs")
    parser.add_argument("--min-date", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-mfs", type=int, default=2)
    return parser


def print_mode_line(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<15} DA={metrics['DA']:.2f}% | "
        f"R2={metrics['R2']:.4f} | MAPE={metrics['MAPE']:.3f}% | MAE={metrics['MAE']:.4f}"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = pd.read_csv(args.data_path)
    df_feat = base.prepare_ohlc_features(data, min_date=args.min_date)
    cfg = base.TrialConfig(
        min_date=args.min_date,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        n_mfs=int(args.n_mfs),
    )
    result = run_branch_ablation(df_feat, cfg)

    out_json = os.path.join(args.output_dir, f"{args.stock}_branch_ablation.json")
    with open(out_json, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    print("=" * 72)
    print("BRANCH ABLATION")
    print("=" * 72)
    print(f"Data: {args.data_path}")
    print(f"Config: min_date={args.min_date}, train_ratio={args.train_ratio:.2f}, val_ratio={args.val_ratio:.2f}, seed={args.seed}")
    print(f"Split: {result['split']}")
    print(f"Train-only typical abs log-return: {result['train_statistics']['typical_abs_close_log_return']:.6f}")
    print("\nValidation selection metrics")
    print_mode_line("full", result["modes"]["full"]["selection_metrics"])
    print_mode_line("price_only", result["modes"]["price_only"]["selection_metrics"])
    print_mode_line("direction_only", result["modes"]["direction_only"]["selection_metrics"])
    print("\nTest metrics")
    print_mode_line("full", result["modes"]["full"]["test_metrics"])
    print_mode_line("price_only", result["modes"]["price_only"]["test_metrics"])
    print_mode_line("direction_only", result["modes"]["direction_only"]["test_metrics"])
    print("\nBaselines")
    print_mode_line("persistence", result["naive_baselines"]["close_persistence"])
    print_mode_line("trend", result["naive_baselines"]["trend_persistence"])
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
