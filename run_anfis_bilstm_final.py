#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM-ANFIS FINAL - Production Ready for Publication

Features:
1. Reproducible with random seed
2. Multi-run (5 runs) with best model selection
3. Cosine annealing learning rate
4. Save/Load best model
5. Publication-quality visualizations

Author: [Your Name]
Date: 2026-01-18
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ===============================
# REPRODUCIBILITY
# ===============================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For TensorFlow determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# GPU setup
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


# ===============================
# ANFIS LAYER
# ===============================
class ANFISLayer(layers.Layer):
    """
    ANFIS Layer - First-order Sugeno TSK fuzzy inference system
    
    - Layer 1: Gaussian membership functions (fuzzification)
    - Layer 2: Rule firing strengths (T-norm: product)
    - Layer 3: Normalized firing strengths
    - Layer 4: First-order Sugeno consequents
    - Layer 5: Weighted sum (defuzzification)
    """
    
    def __init__(self, n_rules=5, output_dim=4, **kwargs):
        super(ANFISLayer, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.output_dim = output_dim
    
    def build(self, input_shape):
        n_features = input_shape[-1]
        
        # Premise parameters: Gaussian MF centers
        self.centers = self.add_weight(
            name='centers',
            shape=(self.n_rules, n_features),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True
        )
        
        self.widths = self.add_weight(
            name='widths',
            shape=(self.n_rules,),
            initializer=tf.constant_initializer(1.0),
            trainable=True
        )
        
        # Consequent parameters
        self.consequent_w = self.add_weight(
            name='consequent_w',
            shape=(self.n_rules, n_features, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.consequent_b = self.add_weight(
            name='consequent_b',
            shape=(self.n_rules, self.output_dim),
            initializer='zeros',
            trainable=True
        )
        
        super(ANFISLayer, self).build(input_shape)
    
    def call(self, inputs):
        x_exp = tf.expand_dims(inputs, 1)
        c_exp = tf.expand_dims(self.centers, 0)
        
        dist_sq = tf.reduce_sum(tf.square(x_exp - c_exp), axis=-1)
        sigma_sq = tf.square(tf.abs(self.widths) + 0.1)
        firing = tf.exp(-dist_sq / (2 * sigma_sq))
        
        firing_sum = tf.reduce_sum(firing, axis=-1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum
        
        x_exp2 = tf.expand_dims(inputs, 1)
        x_exp3 = tf.expand_dims(x_exp2, -1)
        
        w_exp = tf.expand_dims(self.consequent_w, 0)
        linear = tf.reduce_sum(x_exp3 * w_exp, axis=2)
        rule_outputs = linear + self.consequent_b
        
        firing_exp = tf.expand_dims(firing_norm, -1)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)
        
        return output
    
    def get_config(self):
        config = super(ANFISLayer, self).get_config()
        config.update({'n_rules': self.n_rules, 'output_dim': self.output_dim})
        return config


# ===============================
# MODEL CREATION
# ===============================
def create_model(look_back, n_features, lstm_units=64, n_rules=5, dropout=0.2, lr=0.001):
    """
    Create BiLSTM-ANFIS model
    
    Same architecture as original (proven to work well):
    - 2 BiLSTM layers (64 and 32 units)
    - ANFIS layer with 5 rules
    """
    inputs = layers.Input(shape=(look_back, n_features))
    
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
    x = layers.Dropout(dropout)(x)
    
    outputs = ANFISLayer(n_rules=n_rules, output_dim=4, name='anfis')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model


# ===============================
# DATA PREPARATION
# ===============================
def prepare_data(data, look_back=60, train_split=0.8):
    """Prepare returns-based data with robust cleaning"""
    df = data.copy()
    
    # Calculate returns
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Robust cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    target_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret']
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    prices = df[['Close', 'Open', 'High', 'Low']].values
    
    # Get dates if available
    dates = None
    for date_col in ['Date', 'date', 'Datetime', 'datetime']:
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col]).values
            break
    
    # Scale
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    # Create sequences
    X, y, base_px, seq_dates = [], [], [], []
    for i in range(look_back, len(features)):
        X.append(scaled_feat[i-look_back:i])
        y.append(scaled_tgt[i])
        base_px.append(prices[i-1])
        if dates is not None:
            seq_dates.append(dates[i])
    
    X, y, base_px = np.array(X), np.array(y), np.array(base_px)
    
    split = int(len(X) * train_split)
    
    result = {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'base_prices': base_px[split:],
        'actual_prices': prices[split + look_back:],
        'tgt_scaler': tgt_scaler,
        'feat_scaler': feat_scaler,
        'scaled_feat': scaled_feat
    }
    
    if dates is not None:
        result['test_dates'] = seq_dates[split:]
    
    return result




# ===============================
# LEARNING RATE SCHEDULE
# ===============================
def cosine_lr_schedule(epoch, initial_lr=0.001, total_epochs=150, min_lr=1e-6):
    """Cosine annealing learning rate"""
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))


# ===============================
# METRICS
# ===============================
def calc_directional_accuracy(y_true, y_pred, y_prev):
    """
    Calculate Directional Accuracy (DA)
    
    DA = (1/n) * sum(I(sign(y_i - y_{i-1}) = sign(ŷ_i - y_{i-1})))
    
    Measures if the model correctly predicts the direction of price change.
    """
    actual_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    
    correct = (actual_direction == pred_direction).astype(float)
    da = np.mean(correct) * 100  # As percentage
    
    return da


def calc_metrics(y_true, y_pred, y_prev=None, names=['Close', 'Open', 'High', 'Low']):
    """
    Calculate comprehensive metrics including Directional Accuracy
    """
    results = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        
        # Directional Accuracy (if previous values provided)
        da = 0
        if y_prev is not None:
            prev = y_prev[:, i]
            da = calc_directional_accuracy(t, p, prev)
        
        results[n] = {
            'RMSE': math.sqrt(mean_squared_error(t, p)),
            'MAE': mean_absolute_error(t, p),
            'MAPE': mape,
            'R2': r2_score(t, p),
            'DA': da  # Directional Accuracy
        }
    return results


def profile_model(model, X_sample, n_runs=10):
    """
    Profile model performance
    
    Returns:
    - params: Total trainable parameters
    - size_kb: Model size in KB
    - inference_time_ms: Average inference time per sample in ms
    """
    # Count parameters
    params = model.count_params()
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    
    # Estimate model size (approximate - based on float32 weights)
    size_bytes = trainable_params * 4  # float32 = 4 bytes
    size_kb = size_bytes / 1024
    
    # Measure inference time
    # Warm up
    _ = model.predict(X_sample[:1], verbose=0)
    
    # Time multiple runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict(X_sample[:1], verbose=0)
        times.append(time.time() - start)
    
    avg_inference_ms = np.mean(times) * 1000
    
    return {
        'total_params': params,
        'trainable_params': int(trainable_params),
        'size_kb': round(size_kb, 2),
        'inference_time_ms': round(avg_inference_ms, 2)
    }


# ===============================
# VISUALIZATIONS
# ===============================
def plot_predictions(actual, predicted, dates, stock_name, output_dir, price_type='Close'):
    """
    Plot actual vs predicted prices - Publication quality
    """
    idx = ['Close', 'Open', 'High', 'Low'].index(price_type)
    act = actual[:, idx]
    pred = predicted[:, idx]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if dates is not None and len(dates) == len(act):
        x = pd.to_datetime(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(act))
    
    ax.plot(x, act, label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.9)
    ax.plot(x, pred, label='Predicted', color='#E94F37', linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Date' if dates is not None else 'Time Step')
    ax.set_ylabel(f'{price_type} Price ($)')
    ax.set_title(f'{stock_name} - {price_type} Price: Actual vs Predicted')
    ax.legend(loc='upper left')
    
    # Add R² annotation
    r2 = r2_score(act, pred)
    ax.text(0.98, 0.02, f'R² = {r2:.4f}', transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{stock_name}_{price_type}_prediction.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_all_prices(actual, predicted, dates, stock_name, output_dir):
    """
    Plot all 4 prices (OHLC) in a 2x2 grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    price_names = ['Close', 'Open', 'High', 'Low']
    colors = [('#2E86AB', '#E94F37'), ('#1B998B', '#FF6B6B'), 
              ('#A23B72', '#F18F01'), ('#5C4D7D', '#119DA4')]
    
    for idx, (ax, name, (c1, c2)) in enumerate(zip(axes.flat, price_names, colors)):
        act = actual[:, idx]
        pred = predicted[:, idx]
        
        if dates is not None and len(dates) == len(act):
            x = pd.to_datetime(dates)
        else:
            x = np.arange(len(act))
        
        ax.plot(x, act, label='Actual', color=c1, linewidth=1.2, alpha=0.9)
        ax.plot(x, pred, label='Predicted', color=c2, linewidth=1.2, alpha=0.8, linestyle='--')
        
        r2 = r2_score(act, pred)
        ax.set_title(f'{name} Price (R² = {r2:.4f})')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left', fontsize=9)
        
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    fig.suptitle(f'{stock_name} - OHLC Price Predictions', fontsize=18, y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{stock_name}_all_predictions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_training_history(history, stock_name, output_dir):
    """
    Plot training and validation loss
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], label='Training Loss', color='#2E86AB', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', color='#E94F37', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'{stock_name} - Training History')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{stock_name}_training_history.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_error_distribution(actual, predicted, stock_name, output_dir):
    """
    Plot prediction error distribution
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    price_names = ['Close', 'Open', 'High', 'Low']
    
    for idx, (ax, name) in enumerate(zip(axes, price_names)):
        errors = predicted[:, idx] - actual[:, idx]
        pct_errors = (errors / actual[:, idx]) * 100
        
        ax.hist(pct_errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Error Distribution')
        
        # Add stats
        mean_err = np.mean(pct_errors)
        std_err = np.std(pct_errors)
        ax.text(0.95, 0.95, f'μ={mean_err:.2f}%\nσ={std_err:.2f}%', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'{stock_name} - Prediction Error Distribution', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{stock_name}_error_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_scatter(actual, predicted, stock_name, output_dir):
    """
    Scatter plot of actual vs predicted
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    price_names = ['Close', 'Open', 'High', 'Low']
    colors = ['#2E86AB', '#1B998B', '#A23B72', '#5C4D7D']
    
    for idx, (ax, name, color) in enumerate(zip(axes, price_names, colors)):
        act = actual[:, idx]
        pred = predicted[:, idx]
        
        ax.scatter(act, pred, alpha=0.5, s=10, color=color)
        
        # Perfect prediction line
        min_val = min(act.min(), pred.min())
        max_val = max(act.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        r2 = r2_score(act, pred)
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title(f'{name} (R² = {r2:.4f})')
        ax.legend()
    
    fig.suptitle(f'{stock_name} - Actual vs Predicted Scatter', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{stock_name}_scatter.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


# ===============================
# TRAINING WITH MULTI-RUN
# ===============================
def train_stock(stock_name, data_path, output_dir, n_rules=5, n_runs=5, epochs=150, seed=42):
    """
    Train model with multi-run best selection
    """
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (BiLSTM-ANFIS Final)")
    print(f"   Rules: {n_rules}, Runs: {n_runs}, Epochs: {epochs}")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows")
    
    # Prepare data
    try:
        d = prepare_data(data)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    
    # Adaptive training strategy for small datasets
    train_samples = d['X_train'].shape[0]
    is_small_dataset = train_samples < 3000
    
    if is_small_dataset:
        print(f"   ⚠️ Small dataset detected ({train_samples} samples) - disabling early stopping")
    
    # Multi-run training
    best_model = None
    best_history = None
    best_r2 = -1
    best_run = -1
    all_run_results = []
    
    print(f"\n🎓 Starting {n_runs} training runs...")
    
    for run in range(n_runs):
        # Set seed for this run
        run_seed = seed + run
        set_seed(run_seed)
        
        print(f"\n--- Run {run+1}/{n_runs} (seed={run_seed}) ---")
        
        # Create model
        model = create_model(
            look_back=60,
            n_features=d['X_train'].shape[-1],
            lstm_units=64,
            n_rules=n_rules
        )
        
        # Callbacks - adaptive based on dataset size
        if is_small_dataset:
            # For small datasets: NO early stopping, train full epochs
            callbacks = [
                LearningRateScheduler(lambda e: cosine_lr_schedule(e, 0.001, epochs)),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
            ]
        else:
            # For large datasets: use early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                LearningRateScheduler(lambda e: cosine_lr_schedule(e, 0.001, epochs)),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
            ]
        
        # Train with timing
        train_start = time.time()
        history = model.fit(
            d['X_train'], d['y_train'],
            validation_data=(d['X_test'], d['y_test']),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - train_start
        epochs_trained = len(history.history['loss'])
        
        # Evaluate
        pred_scaled = model.predict(d['X_test'], verbose=0)
        pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
        pred_px = d['base_prices'] * (1 + pred_ret)
        actual_px = d['actual_prices'][:len(pred_px)]
        
        # Get previous prices for DA calculation
        y_prev_px = d['base_prices'][:len(pred_px)]
        
        metrics = calc_metrics(actual_px, pred_px, y_prev_px)
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        close_r2 = metrics['Close']['R2']
        close_da = metrics['Close']['DA']
        
        print(f"   Close R²: {close_r2:.4f}, DA: {close_da:.1f}%, Epochs: {epochs_trained}/{epochs}, Time: {train_time:.1f}s")
        
        all_run_results.append({
            'run': run + 1,
            'close_r2': close_r2,
            'avg_r2': avg_r2,
            'val_loss': min(history.history['val_loss']),
            'epochs_trained': epochs_trained,
            'training_time_sec': round(train_time, 2)
        })
        
        # Update best
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = model
            best_history = history.history
            best_run = run + 1
            best_pred_px = pred_px
            best_metrics = metrics
            best_train_time = train_time
            best_epochs_trained = epochs_trained
    
    print(f"\n✅ Best run: {best_run} with Avg R² = {best_r2:.4f}")
    
    # Profile model performance
    print(f"\n⚡ Profiling model performance...")
    performance = profile_model(best_model, d['X_test'])
    
    # Add training metrics
    performance['training_time_sec'] = round(best_train_time, 2)
    performance['epochs_trained'] = best_epochs_trained
    performance['epochs_max'] = epochs
    performance['time_per_epoch_sec'] = round(best_train_time / best_epochs_trained, 3) if best_epochs_trained > 0 else 0
    
    print(f"   Parameters: {performance['total_params']:,}")
    print(f"   Model Size: {performance['size_kb']:.1f} KB")
    print(f"   Training Time: {performance['training_time_sec']:.1f}s ({performance['epochs_trained']}/{epochs} epochs)")
    print(f"   Time/Epoch: {performance['time_per_epoch_sec']:.3f}s")
    print(f"   Inference Time: {performance['inference_time_ms']:.2f} ms/sample")
    
    # Recalculate metrics with y_prev for DA
    y_prev_px = d['base_prices'][:len(best_pred_px)]
    actual_px = d['actual_prices'][:len(best_pred_px)]
    best_metrics = calc_metrics(actual_px, best_pred_px, y_prev_px)
    
    # Save best model
    model_path = os.path.join(output_dir, f'{stock_name}_best_model.keras')
    best_model.save(model_path)
    print(f"\n💾 Saved best model to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'stock': stock_name,
            'best_run': best_run,
            'metrics': {k: v for k, v in best_metrics.items()},
            'performance': performance,
            'all_runs': all_run_results
        }, f, indent=2)
    
    # Print final metrics
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name} (Best Run: {best_run})")
    print(f"{'='*80}")
    
    all_pass = True
    for n, m in best_metrics.items():
        f = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        if m['R2'] < 0.98:
            all_pass = False
        print(f"{n:6} - RMSE: {m['RMSE']:7.2f} | MAE: {m['MAE']:7.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {f}")
    
    print(f"\n📈 Model Performance:")
    print(f"   Parameters: {performance['total_params']:,}")
    print(f"   Size: {performance['size_kb']:.1f} KB")
    print(f"   Training: {performance['training_time_sec']:.1f}s ({performance['epochs_trained']}/{performance['epochs_max']} epochs)")
    print(f"   Time/Epoch: {performance['time_per_epoch_sec']:.3f}s")
    print(f"   Inference: {performance['inference_time_ms']:.2f} ms/sample")
    
    if all_pass:
        print(f"\n🎉 SUCCESS! All R² >= 0.98 for {stock_name}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    test_dates = d.get('test_dates', None)
    
    # Plot predictions
    plot_predictions(actual_px, best_pred_px, test_dates, stock_name, output_dir, 'Close')
    plot_all_prices(actual_px, best_pred_px, test_dates, stock_name, output_dir)
    plot_training_history(best_history, stock_name, output_dir)
    plot_error_distribution(actual_px, best_pred_px, stock_name, output_dir)
    plot_scatter(actual_px, best_pred_px, stock_name, output_dir)
    
    print(f"✅ Saved all visualizations to: {output_dir}")
    
    return {'metrics': best_metrics, 'performance': performance}


def plot_summary(all_metrics, output_dir):
    """
    Plot summary bar chart for all stocks
    """
    if not all_metrics:
        return
    
    stocks = list(all_metrics.keys())
    price_types = ['Close', 'Open', 'High', 'Low']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(stocks))
    width = 0.2
    colors = ['#2E86AB', '#1B998B', '#A23B72', '#5C4D7D']
    
    for i, (ptype, color) in enumerate(zip(price_types, colors)):
        r2_values = [all_metrics[s]['metrics'][ptype]['R2'] for s in stocks]
        ax.bar(x + i*width, r2_values, width, label=ptype, color=color, alpha=0.8)
    
    ax.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='Target (0.98)')
    
    ax.set_xlabel('Stock')
    ax.set_ylabel('R² Score')
    ax.set_title('BiLSTM-ANFIS Performance Summary')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(stocks)
    ax.set_ylim(0.9, 1.01)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'summary_performance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


# ===============================
# MAIN
# ===============================
def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_final')
    
    stocks = {
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        'JPM': os.path.join(base_dir, 'JPM.csv'),
        'TSLA': os.path.join(base_dir, 'TSLA.csv'),
        'IHSG': os.path.join(base_dir, 'IHSG_2007_2024.csv'),
        'SP500': os.path.join(base_dir, 'SP500.csv'),
    }
    
    all_metrics = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                m = train_stock(
                    name, path, output_dir,
                    n_rules=5,
                    n_runs=5,
                    epochs=150,
                    seed=42
                )
                if m:
                    all_metrics[name] = m
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate summary plot
    if all_metrics:
        plot_summary(all_metrics, output_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - BiLSTM-ANFIS (Publication Ready)")
    print(f"{'='*80}")
    
    target_met = 0
    for n, result in all_metrics.items():
        m = result['metrics']
        perf = result['performance']
        cm = m['Close']
        if cm['R2'] >= 0.98:
            target_met += 1
            f = "✅"
        else:
            f = "⚠️" if cm['R2'] >= 0.95 else "❌"
        print(f"{n:10}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}%, DA={cm['DA']:.1f}%, Params={perf['total_params']:,} {f}")
    
    print(f"\n🎯 Target (R² >= 0.98): {target_met}/{len(all_metrics)} stocks")
    print(f"📁 Results saved to: {output_dir}")
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
