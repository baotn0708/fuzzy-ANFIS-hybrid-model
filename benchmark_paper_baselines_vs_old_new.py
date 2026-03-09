#!/usr/bin/env python3
"""Paper-style benchmark: original ANFIS vs common baselines vs newest exogenous hybrid.

Design:
- Original model and paper baselines use only OHLC-derived features.
- Newest model uses the fuller exogenous dataset/pipeline.
- All models share the same raw calendar split dates.
- Classical baselines tune hyperparameters on validation, then refit on train+val.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

import benchmark_models as paper_models
from benchmark_original_vs_best import (
    TARGETS,
    aggregate_model_runs,
    compute_price_metrics,
    compute_shared_split_dates,
    ensure_dir,
    load_shared_source,
    measure_inference_ms_per_sample,
    prepare_original_data_by_dates,
    serialize,
    summary_rows_for_stock,
)
from benchmark_original_vs_reasoned_exog import HybridConfig, train_original_once, train_reasoned_once

DEFAULT_MODELS = [
    "original",
    "arima",
    "svr",
    "knn",
    "random_forest",
    "xgboost",
    "mlp",
    "lstm",
    "gru",
    "cnn_lstm",
    "transformer",
]

MODEL_LABELS = {
    "original": "Original Feature-Group ANFIS + BiLSTM",
    "arima": "ARIMA",
    "svr": "SVR",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "Transformer",
    "reasoned_exog": "Reasoned Exogenous ANFIS Hybrid",
}


# Reduce TensorFlow log noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Feature-Group ANFIS + BiLSTM vs common baselines on AMZN/JPM/TSLA")
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--stocks", type=str, default="AMZN,JPM,TSLA", help="Comma-separated stock keys")
    p.add_argument("--stock-starts", type=str, default="AMZN:1997-05-15,JPM:1980-03-17,TSLA:2010-06-29")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma-separated model keys")
    p.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    p.add_argument("--epochs", type=int, default=12, help="Max epochs for DL baselines and new hybrid")
    p.add_argument("--original-epochs", type=int, default=150, help="Max epochs for original model")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--look-back", type=int, default=60)
    p.add_argument("--top-exog-features", type=int, default=10)
    p.add_argument("--recency-beta", type=float, default=0.20)
    p.add_argument("--source-mode", type=str, choices=["raw", "with_exog", "auto"], default="raw")
    p.add_argument("--output-dir", type=str, default="benchmark_common_baselines_three_stocks")
    return p.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_stock_starts(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in parse_csv_list(raw):
        if ":" not in token:
            raise ValueError(f"Invalid STOCK:DATE token: {token}")
        stock, start = token.split(":", 1)
        out[stock.strip().upper()] = start.strip()
    return out


def shared_stock_paths(base_dir: str, source_mode: str) -> Dict[str, str]:
    def choose(raw_name: str, exog_name: str) -> str:
        raw_path = os.path.join(base_dir, "Dataset", raw_name)
        exog_path = os.path.join(base_dir, "Dataset", exog_name)
        if source_mode == "raw":
            return raw_path
        if source_mode == "with_exog":
            return exog_path
        if os.path.exists(raw_path):
            return raw_path
        return exog_path

    return {
        "AMZN": choose("AMZN.csv", "AMZN_with_exog.csv"),
        "JPM": choose("JPM.csv", "JPM_with_exog.csv"),
        "TSLA": choose("TSLA.csv", "TSLA_with_exog.csv"),
    }


def utility_score(metrics: Dict[str, object]) -> float:
    return (
        0.58 * float(metrics["Close"]["DA"])
        + 0.08 * float(metrics["Model"]["Avg_DA"])
        + 12.0 * float(metrics["Close"]["R2"])
        - 0.10 * float(metrics["Close"]["MAPE"])
        - 120.0 * max(0.0, 0.95 - float(metrics["Close"]["R2"]))
    )


class SequenceValidationUtility(tf.keras.callbacks.Callback):
    def __init__(self, d: Dict[str, object]):
        super().__init__()
        self.d = d
        self.best_score = -1e18
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        pred_scaled = self.model.predict(self.d["X_val"], verbose=0)
        pred_ret = self.d["tgt_scaler"].inverse_transform(np.asarray(pred_scaled, dtype=np.float32))
        pred_px = decode_prices(self.d["curr_val"], pred_ret)
        metrics = compute_price_metrics(self.d["next_val"], pred_px, self.d["curr_val"])
        score = utility_score(metrics)
        if score > self.best_score:
            self.best_score = float(score)
            self.best_epoch = int(epoch) + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def decode_prices(curr: np.ndarray, pred_ret: np.ndarray) -> np.ndarray:
    return curr[:, 0:1] * (1.0 + pred_ret)


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], int(X.shape[1] * X.shape[2]))).astype(np.float32)


def measure_sklearn_inference_ms_per_sample(model, X_test: np.ndarray, repeats: int = 5) -> float:
    if len(X_test) == 0:
        return 0.0
    _ = model.predict(X_test[: min(len(X_test), 64)])
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings) * 1000.0 / len(X_test))


def clone_original_split_with_full_train(d: Dict[str, object]) -> Dict[str, object]:
    out = dict(d)
    out["X_fit"] = np.concatenate([d["X_train"], d["X_val"]], axis=0)
    out["y_fit"] = np.concatenate([d["y_train"], d["y_val"]], axis=0)
    out["curr_fit"] = np.concatenate([d["curr_train"], d["curr_val"]], axis=0)
    out["next_fit"] = np.concatenate([d["next_train"], d["next_val"]], axis=0)
    return out


def build_keras_result(
    model_key: str,
    model_name: str,
    model,
    d: Dict[str, object],
    pred_ret_test: np.ndarray,
    train_time: float,
    epochs_trained: int,
    selected_epoch: Optional[int],
    seed: int,
) -> Dict[str, object]:
    pred_px = decode_prices(d["curr_test"], pred_ret_test)
    metrics = compute_price_metrics(d["next_test"], pred_px, d["curr_test"])
    infer_ms = measure_inference_ms_per_sample(model, d["X_test"])
    result = {
        "model_key": model_key,
        "model_name": model_name,
        "seed": int(seed),
        "architecture": {
            "look_back": int(d["look_back"]),
            "n_features": int(d["X_train"].shape[-1]),
            "feature_cols": d["feature_cols"],
            "outputs": ["next_close_ret", "next_open_ret", "next_high_ret", "next_low_ret"],
        },
        "split_info": {
            "train_sequences": int(len(d["X_train"])),
            "val_sequences": int(len(d["X_val"])),
            "test_sequences": int(len(d["X_test"])),
            "train_sample_start": d["date_train"][0].strftime("%Y-%m-%d"),
            "train_sample_end": d["date_train"][-1].strftime("%Y-%m-%d"),
            "val_sample_start": d["date_val"][0].strftime("%Y-%m-%d"),
            "val_sample_end": d["date_val"][-1].strftime("%Y-%m-%d"),
            "test_sample_start": d["date_test"][0].strftime("%Y-%m-%d"),
            "test_sample_end": d["date_test"][-1].strftime("%Y-%m-%d"),
        },
        "timing": {
            "training_time_sec": float(train_time),
            "epochs_trained": int(epochs_trained),
            "selected_epoch": int(selected_epoch or epochs_trained),
            "inference_ms_per_sample": float(infer_ms),
        },
        "params": int(model.count_params()),
        "metrics": metrics,
    }
    return result


def train_sequence_baseline_once(
    model_key: str,
    d: Dict[str, object],
    seed: int,
    epochs: int,
    builder: Callable[[int, int], tf.keras.Model],
) -> Dict[str, object]:
    set_seed(seed)
    tf.keras.backend.clear_session()
    K.clear_session()

    model = builder(d["look_back"], int(d["X_train"].shape[-1]))
    utility_cb = SequenceValidationUtility(d)
    if hasattr(paper_models, "original_lr_schedule"):
        lr_sched = lambda e: paper_models.original_lr_schedule(e, initial_lr=0.001, total_epochs=max(epochs, 1), min_lr=1e-5)
    else:
        lr_sched = lambda e: 0.001
    callbacks = [
        utility_cb,
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=False),
        LearningRateScheduler(lr_sched),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0),
    ]

    t0 = time.time()
    history = model.fit(
        d["X_train"],
        d["y_train"],
        validation_data=(d["X_val"], d["y_val"]),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - t0
    pred_scaled = model.predict(d["X_test"], verbose=0)
    pred_ret_test = d["tgt_scaler"].inverse_transform(np.asarray(pred_scaled, dtype=np.float32))
    return build_keras_result(
        model_key=model_key,
        model_name=MODEL_LABELS[model_key],
        model=model,
        d=d,
        pred_ret_test=pred_ret_test,
        train_time=train_time,
        epochs_trained=len(history.history.get("loss", [])),
        selected_epoch=utility_cb.best_epoch,
        seed=seed,
    )


def fit_classical_model(builder: Callable[[], object], X_train: np.ndarray, y_train: np.ndarray):
    model = builder()
    model.fit(X_train, y_train)
    return model


def classical_result(
    model_key: str,
    d: Dict[str, object],
    seed: int,
    model,
    pred_ret_test: np.ndarray,
    train_time: float,
    extra_arch: Optional[Dict[str, object]] = None,
    params: Optional[int] = None,
) -> Dict[str, object]:
    pred_px = decode_prices(d["curr_test"], pred_ret_test)
    metrics = compute_price_metrics(d["next_test"], pred_px, d["curr_test"])
    infer_ms = measure_sklearn_inference_ms_per_sample(model, flatten_sequences(d["X_test"]))
    arch = {
        "look_back": int(d["look_back"]),
        "n_features": int(d["X_train"].shape[-1]),
        "feature_cols": d["feature_cols"],
        "outputs": ["next_close_ret", "next_open_ret", "next_high_ret", "next_low_ret"],
    }
    if extra_arch:
        arch.update(extra_arch)
    return {
        "model_key": model_key,
        "model_name": MODEL_LABELS[model_key],
        "seed": int(seed),
        "architecture": arch,
        "split_info": {
            "train_sequences": int(len(d["X_train"])),
            "val_sequences": int(len(d["X_val"])),
            "test_sequences": int(len(d["X_test"])),
            "train_sample_start": d["date_train"][0].strftime("%Y-%m-%d"),
            "train_sample_end": d["date_train"][-1].strftime("%Y-%m-%d"),
            "val_sample_start": d["date_val"][0].strftime("%Y-%m-%d"),
            "val_sample_end": d["date_val"][-1].strftime("%Y-%m-%d"),
            "test_sample_start": d["date_test"][0].strftime("%Y-%m-%d"),
            "test_sample_end": d["date_test"][-1].strftime("%Y-%m-%d"),
        },
        "timing": {
            "training_time_sec": float(train_time),
            "epochs_trained": 0,
            "selected_epoch": 0,
            "inference_ms_per_sample": float(infer_ms),
        },
        "params": int(params if params is not None else -1),
        "metrics": metrics,
    }


def train_flat_baseline_once(
    model_key: str,
    d: Dict[str, object],
    seed: int,
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]],
) -> Dict[str, object]:
    total_t0 = time.time()
    X_train = flatten_sequences(d["X_train"])
    X_val = flatten_sequences(d["X_val"])
    X_fit = flatten_sequences(np.concatenate([d["X_train"], d["X_val"]], axis=0))
    X_test = flatten_sequences(d["X_test"])
    y_fit = np.concatenate([d["y_train"], d["y_val"]], axis=0)

    best = None
    for cfg, builder in candidates:
        t0 = time.time()
        model = fit_classical_model(lambda b=builder, c=cfg: b(c), X_train, d["y_train"])
        train_time = time.time() - t0
        pred_val_scaled = np.asarray(model.predict(X_val), dtype=np.float32)
        pred_val_ret = d["tgt_scaler"].inverse_transform(pred_val_scaled)
        pred_val_px = decode_prices(d["curr_val"], pred_val_ret)
        metrics = compute_price_metrics(d["next_val"], pred_val_px, d["curr_val"])
        score = utility_score(metrics)
        if best is None or score > best["score"]:
            best = {
                "cfg": cfg,
                "builder": builder,
                "score": float(score),
                "val_metrics": metrics,
                "train_time": float(train_time),
            }

    model = fit_classical_model(lambda c=best["cfg"], b=best["builder"]: b(c), X_fit, y_fit)
    train_time = time.time() - total_t0
    pred_test_scaled = np.asarray(model.predict(X_test), dtype=np.float32)
    pred_test_ret = d["tgt_scaler"].inverse_transform(pred_test_scaled)
    params = -1
    extra_arch = {"input_format": "flattened_sequence", "selected_hyperparams": best["cfg"]}
    return classical_result(model_key, clone_original_split_with_full_train(d), seed, model, pred_test_ret, train_time, extra_arch=extra_arch, params=params)


def train_svr_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]] = []
    for C in [5.0, 10.0]:
        for eps in [0.01, 0.03]:
            cfg = {"C": C, "epsilon": eps, "kernel": "rbf", "gamma": "scale"}
            candidates.append((cfg, lambda c: MultiOutputRegressor(SVR(C=c["C"], epsilon=c["epsilon"], kernel=c["kernel"], gamma=c["gamma"]))))
    return train_flat_baseline_once("svr", d, seed, candidates)


def train_random_forest_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]] = []
    for n_estimators, max_depth, min_samples_leaf in [(300, 8, 2), (500, None, 2), (500, 12, 1)]:
        cfg = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
        candidates.append((cfg, lambda c, s=seed: RandomForestRegressor(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            min_samples_leaf=c["min_samples_leaf"],
            random_state=s,
            n_jobs=-1,
        )))
    return train_flat_baseline_once("random_forest", d, seed, candidates)


def train_knn_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]] = []
    for n_neighbors, weights in [(10, "distance"), (20, "distance"), (30, "uniform")]:
        cfg = {"n_neighbors": n_neighbors, "weights": weights}
        candidates.append((cfg, lambda c: KNeighborsRegressor(n_neighbors=c["n_neighbors"], weights=c["weights"])))
    return train_flat_baseline_once("knn", d, seed, candidates)


def train_xgboost_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]] = []
    for n_estimators, max_depth, learning_rate in [(300, 4, 0.05), (500, 4, 0.03)]:
        cfg = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
        candidates.append((cfg, lambda c, s=seed: MultiOutputRegressor(XGBRegressor(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=s,
            n_jobs=4,
            verbosity=0,
        ))))
    return train_flat_baseline_once("xgboost", d, seed, candidates)


def train_mlp_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    candidates: List[Tuple[Dict[str, object], Callable[[Dict[str, object]], object]]] = []
    for hidden, alpha in [((128, 64), 1e-4), ((256, 128), 1e-4), ((128, 64), 5e-4)]:
        cfg = {"hidden_layer_sizes": hidden, "alpha": alpha}
        candidates.append((cfg, lambda c, s=seed: MLPRegressor(
            hidden_layer_sizes=c["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=c["alpha"],
            learning_rate_init=5e-4,
            max_iter=300,
            random_state=s,
            early_stopping=True,
            validation_fraction=0.1,
        )))
    return train_flat_baseline_once("mlp", d, seed, candidates)


def arima_roll_forecast(train_series: np.ndarray, eval_series: np.ndarray, order: Tuple[int, int, int]) -> Tuple[np.ndarray, int, float]:
    if ARIMA is None:
        raise RuntimeError("statsmodels is not installed")
    history = np.asarray(train_series, dtype=np.float64)
    t0 = time.time()
    fit = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    fit_time = time.time() - t0
    preds = []
    for actual in np.asarray(eval_series, dtype=np.float64):
        forecast = fit.forecast(steps=1)
        preds.append(float(np.asarray(forecast)[0]))
        try:
            fit = fit.append([float(actual)], refit=False)
        except Exception:
            history = np.append(history, float(actual))
            fit = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    n_params = int(len(np.asarray(getattr(fit, "params", []), dtype=np.float32)))
    return np.asarray(preds, dtype=np.float32), n_params, fit_time


def choose_arima_order(d: Dict[str, object]) -> Tuple[int, int, int]:
    y_train_raw = d["tgt_scaler"].inverse_transform(d["y_train"])[:, 0]
    y_val_raw = d["tgt_scaler"].inverse_transform(d["y_val"])[:, 0]
    candidates = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (3, 0, 1)]
    best = None
    for order in candidates:
        try:
            pred_val_close, _, _ = arima_roll_forecast(y_train_raw, y_val_raw, order)
            pred_val_ret = np.zeros_like(d["y_val"], dtype=np.float32)
            pred_val_ret[:, 0] = pred_val_close
            pred_val_px = decode_prices(d["curr_val"], pred_val_ret)
            metrics = compute_price_metrics(d["next_val"], pred_val_px, d["curr_val"])
            score = utility_score(metrics)
            if best is None or score > best["score"]:
                best = {"order": order, "score": float(score), "metrics": metrics}
        except Exception:
            continue
    if best is None:
        raise RuntimeError("ARIMA order search failed on all candidates")
    return best["order"]


def train_arima_once(d: Dict[str, object], seed: int) -> Dict[str, object]:
    order = choose_arima_order(d)
    y_fit_raw = np.concatenate([
        d["tgt_scaler"].inverse_transform(d["y_train"]),
        d["tgt_scaler"].inverse_transform(d["y_val"]),
    ], axis=0)
    y_test_raw = d["tgt_scaler"].inverse_transform(d["y_test"])

    pred_test_ret = np.zeros_like(y_test_raw, dtype=np.float32)
    total_fit_time = 0.0
    total_params = 0
    t0 = time.time()
    for j in range(y_fit_raw.shape[1]):
        preds_j, n_params_j, fit_time_j = arima_roll_forecast(y_fit_raw[:, j], y_test_raw[:, j], order)
        pred_test_ret[:, j] = preds_j
        total_fit_time += fit_time_j
        total_params += n_params_j
    train_time = time.time() - t0

    class DummyArimaPredictor:
        def __init__(self, pred):
            self.pred = pred
        def predict(self, X):
            return np.zeros((len(X), self.pred.shape[1]), dtype=np.float32)

    extra_arch = {"input_format": "univariate_return_series", "order": list(order), "selection_target": "Close"}
    result = classical_result(
        "arima",
        clone_original_split_with_full_train(d),
        seed,
        DummyArimaPredictor(pred_test_ret),
        pred_test_ret,
        train_time,
        extra_arch=extra_arch,
        params=total_params,
    )
    result["timing"]["arima_refit_time_sec"] = float(total_fit_time)
    result["timing"]["inference_ms_per_sample"] = 0.0
    return result


def run_model_once(model_key: str, stock: str, df_raw: pd.DataFrame, split_dates: Dict[str, object], seed: int, base_dir: str, args) -> Dict[str, object]:
    if model_key == "original":
        return train_original_once(stock, df_raw, split_dates, seed, args.original_epochs)
    if model_key == "reasoned_exog":
        cfg = HybridConfig(epochs=args.epochs, top_exog_features=args.top_exog_features, recency_beta=args.recency_beta)
        return train_reasoned_once(stock, df_raw, split_dates, seed, cfg, base_dir)

    d = prepare_original_data_by_dates(df_raw, split_dates, look_back=args.look_back)

    if model_key == "arima":
        return train_arima_once(d, seed)
    if model_key == "svr":
        return train_svr_once(d, seed)
    if model_key == "knn":
        return train_knn_once(d, seed)
    if model_key == "random_forest":
        return train_random_forest_once(d, seed)
    if model_key == "xgboost":
        return train_xgboost_once(d, seed)
    if model_key == "mlp":
        return train_mlp_once(d, seed)
    if model_key == "lstm":
        return train_sequence_baseline_once("lstm", d, seed, args.epochs, paper_models.create_pure_lstm)
    if model_key == "gru":
        return train_sequence_baseline_once("gru", d, seed, args.epochs, paper_models.create_gru)
    if model_key == "cnn_lstm":
        return train_sequence_baseline_once("cnn_lstm", d, seed, args.epochs, paper_models.create_cnn_lstm)
    if model_key == "transformer":
        return train_sequence_baseline_once("transformer", d, seed, args.epochs, paper_models.create_transformer)

    raise KeyError(f"Unknown model key: {model_key}")


def benchmark_stock(stock: str, data_path: str, start_date: str, model_keys: List[str], seeds: List[int], out_dir: str, base_dir: str, args) -> Dict[str, object]:
    shared_df = load_shared_source(data_path, start_date)
    split_dates = compute_shared_split_dates(shared_df, args.train_ratio, args.val_ratio)
    stock_dir = os.path.join(out_dir, stock)
    ensure_dir(stock_dir)

    per_model_runs: Dict[str, List[Dict[str, object]]] = {k: [] for k in model_keys}
    errors: Dict[str, List[Dict[str, object]]] = {k: [] for k in model_keys}

    for model_key in model_keys:
        for seed in seeds:
            print(f"[{stock}] seed={seed} | training {model_key}")
            try:
                result = run_model_once(model_key, stock, shared_df, split_dates, seed, base_dir, args)
                per_model_runs[model_key].append(result)
                with open(os.path.join(stock_dir, f"{stock}_{model_key}_seed{seed}.json"), "w", encoding="utf-8") as f:
                    json.dump(serialize(result), f, indent=2)
                print(
                    f"[{stock}] seed={seed} | {model_key} Close DA={result['metrics']['Close']['DA']:.2f} | "
                    f"R2={result['metrics']['Close']['R2']:.4f} | MAPE={result['metrics']['Close']['MAPE']:.3f}"
                )
            except Exception as exc:
                errors[model_key].append({"seed": int(seed), "error": str(exc)})
                print(f"[{stock}] seed={seed} | {model_key} failed: {exc}")

    models_summary = {}
    for model_key, runs in per_model_runs.items():
        if runs:
            models_summary[model_key] = aggregate_model_runs(runs)

    payload = {
        "stock": stock,
        "data_path": data_path,
        "shared_start_date": start_date,
        "split_dates": split_dates,
        "models": models_summary,
        "errors": {k: v for k, v in errors.items() if v},
    }
    with open(os.path.join(stock_dir, f"{stock}_benchmark_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(serialize(payload), f, indent=2)
    return payload


def write_report(results_by_stock: Dict[str, Dict[str, object]], out_dir: str, seeds: List[int], model_keys: List[str]) -> None:
    has_exog_model = "reasoned_exog" in model_keys
    ohlc_models = [MODEL_LABELS[m] for m in model_keys if m != "reasoned_exog"]
    lines = [
        "# Benchmark: Feature-Group ANFIS + Common Baselines",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- OHLC-only models: {', '.join(ohlc_models)}.",
        "",
    ]
    if has_exog_model:
        lines.insert(5, "- Full-data model: Reasoned Exogenous ANFIS Hybrid.")
    winner_rows = []
    summary_rows = []
    for stock, payload in results_by_stock.items():
        split_dates = payload["split_dates"]
        lines.append(f"## {stock}")
        lines.append("")
        lines.append(f"- Raw window: {split_dates['raw_start_date'].strftime('%Y-%m-%d')} -> {split_dates['raw_end_date'].strftime('%Y-%m-%d')}")
        lines.append(
            f"- Shared cutoffs: train <= {split_dates['train_end_date'].strftime('%Y-%m-%d')}, "
            f"val <= {split_dates['val_end_date'].strftime('%Y-%m-%d')}, test > {split_dates['val_end_date'].strftime('%Y-%m-%d')}"
        )
        lines.append("")
        lines.append("| Model | Input regime | Close DA | Close R2 | Close MAPE | Avg DA | Avg R2 | Params | Train sec |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        stock_models = payload["models"]
        for model_key in model_keys:
            if model_key not in stock_models:
                continue
            model_summary = stock_models[model_key]
            m = model_summary["metrics"]
            input_regime = "OHLC-only" if model_key != "reasoned_exog" else "fuller exogenous"
            params_mean = float(model_summary["params"]["mean"])
            params_std = float(model_summary["params"]["std"])
            params_disp = "N/A" if params_mean < 0 else f"{params_mean:.0f} +/- {params_std:.0f}"
            lines.append(
                f"| {model_summary['model_name']} | {input_regime} | "
                f"{m['Close']['DA']['mean']:.2f} +/- {m['Close']['DA']['std']:.2f} | "
                f"{m['Close']['R2']['mean']:.4f} +/- {m['Close']['R2']['std']:.4f} | "
                f"{m['Close']['MAPE']['mean']:.3f} +/- {m['Close']['MAPE']['std']:.3f} | "
                f"{m['Model']['Avg_DA']['mean']:.2f} +/- {m['Model']['Avg_DA']['std']:.2f} | "
                f"{m['Model']['Avg_R2']['mean']:.4f} +/- {m['Model']['Avg_R2']['std']:.4f} | "
                f"{params_disp} | "
                f"{model_summary['timing']['training_time_sec_mean']:.2f} |"
            )
            summary_rows.append(summary_rows_for_stock(stock, split_dates, model_summary))

        close_da_winner = max(
            ((k, stock_models[k]["metrics"]["Close"]["DA"]["mean"]) for k in stock_models),
            key=lambda x: x[1],
        )
        close_r2_winner = max(
            ((k, stock_models[k]["metrics"]["Close"]["R2"]["mean"]) for k in stock_models),
            key=lambda x: x[1],
        )
        close_mape_winner = min(
            ((k, stock_models[k]["metrics"]["Close"]["MAPE"]["mean"]) for k in stock_models),
            key=lambda x: x[1],
        )
        winner_rows.append({
            "Stock": stock,
            "Close_DA_Winner": MODEL_LABELS[close_da_winner[0]],
            "Close_R2_Winner": MODEL_LABELS[close_r2_winner[0]],
            "Close_MAPE_Winner": MODEL_LABELS[close_mape_winner[0]],
        })
        lines.append("")

    if winner_rows:
        lines.append("## Winners")
        lines.append("")
        winner_df = pd.DataFrame(winner_rows)
        winner_df.to_csv(os.path.join(out_dir, "winner_table.csv"), index=False)
        lines.append(winner_df.to_markdown(index=False))
        lines.append("")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out_dir, "benchmark_summary.csv"), index=False)
        lines.append("## Machine-Readable Summary")
        lines.append("")
        lines.append(summary_df.to_markdown(index=False))
        lines.append("")

    with open(os.path.join(out_dir, "BENCHMARK_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(base_dir, args.output_dir)
    ensure_dir(out_dir)

    model_keys = [m.lower() for m in parse_csv_list(args.models)]
    for model_key in model_keys:
        if model_key not in MODEL_LABELS:
            raise KeyError(f"Unknown model key: {model_key}")
        if model_key == "xgboost" and XGBRegressor is None:
            raise RuntimeError("xgboost requested but not installed")
        if model_key == "arima" and ARIMA is None:
            raise RuntimeError("statsmodels requested but not installed")

    stocks = [s.upper() for s in parse_csv_list(args.stocks)]
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    start_map = parse_stock_starts(args.stock_starts)
    stock_map = shared_stock_paths(base_dir, args.source_mode)

    manifest = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": stocks,
        "models": model_keys,
        "seeds": seeds,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "look_back": args.look_back,
        "source_mode": args.source_mode,
        "paper_baselines": [MODEL_LABELS[m] for m in model_keys if m not in {"original", "reasoned_exog"}],
        "old_model_input": "OHLC-derived 6 features",
    }
    if "reasoned_exog" in model_keys:
        manifest["new_model_input"] = "Fuller exogenous dataset with market/sector/VIX/rates/fundamentals"
    with open(os.path.join(out_dir, "benchmark_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(serialize(manifest), f, indent=2)

    results_by_stock = {}
    errors = {}
    for stock in stocks:
        if stock not in stock_map:
            errors[stock] = {"error": f"Unknown stock key: {stock}"}
            continue
        if stock not in start_map:
            errors[stock] = {"error": f"Missing start date for {stock}"}
            continue
        try:
            results_by_stock[stock] = benchmark_stock(
                stock=stock,
                data_path=stock_map[stock],
                start_date=start_map[stock],
                model_keys=model_keys,
                seeds=seeds,
                out_dir=out_dir,
                base_dir=base_dir,
                args=args,
            )
        except Exception as exc:
            errors[stock] = {"error": str(exc)}

    with open(os.path.join(out_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(serialize(results_by_stock), f, indent=2)
    with open(os.path.join(out_dir, "benchmark_errors.json"), "w", encoding="utf-8") as f:
        json.dump(serialize(errors), f, indent=2)

    if results_by_stock:
        write_report(results_by_stock, out_dir, seeds, model_keys)
    print(f"Artifacts: {out_dir}")
    if errors:
        print(f"Errors: {len(errors)} stock(s) failed. See benchmark_errors.json")


if __name__ == "__main__":
    main()
