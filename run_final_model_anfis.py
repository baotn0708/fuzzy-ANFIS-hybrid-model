#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final ANFIS + BiLSTM Model for Report
=====================================

This script generates the final model for reporting purposes.
It includes:
1. Feature-Group ANFIS + BiLSTM architecture.
2. Comprehensive metric calculation (MAE, MSE, RMSE, R^2, MAPE, Time).
3. Metric persistence to JSON.
4. High-quality visualization of results.
5. Model serialization.

Architecture:
- Returns (4 features) -> ANFIS_returns (2 MFs)
- Indicators (2 features) -> ANFIS_indic (2 MFs)
- BiLSTM -> Temporal features
- Combined -> Dense -> Output
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import time
import warnings
import sys

warnings.filterwarnings('ignore')

# Set plotting style/params for publication quality
plt.style.use('default') 
# plt.style.use('seaborn-v0_8-whitegrid') # Optional: use if available, else stick to default
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# GPU setup
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ===============================
# ANFIS LAYER (Feature-Group)
# ===============================
class FeatureGroupANFIS(layers.Layer):
    """
    ANFIS for a specific feature group.
    Uses 2 Gaussian Membership Functions per feature.
    """
    
    def __init__(self, n_mfs=2, output_dim=4, name_prefix='anfis', initial_centers=None, **kwargs):
        super(FeatureGroupANFIS, self).__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.initial_centers = initial_centers
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        self.n_rules = self.n_mfs ** n_features
        
        if self.initial_centers is not None:
            init_centers = tf.constant_initializer(self.initial_centers)
        else:
            init_centers = tf.keras.initializers.RandomUniform(-1.0, 1.0)
        
        self.mf_centers = self.add_weight(
            name=f'{self.name_prefix}_mf_centers',
            shape=(n_features, self.n_mfs),
            initializer=init_centers,
            trainable=True
        )
        
        self.mf_widths = self.add_weight(
            name=f'{self.name_prefix}_mf_widths',
            shape=(n_features, self.n_mfs),
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )
        
        self.consequent_p = self.add_weight(
            name=f'{self.name_prefix}_consequent_p',
            shape=(self.n_rules, n_features, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.consequent_r = self.add_weight(
            name=f'{self.name_prefix}_consequent_r',
            shape=(self.n_rules, self.output_dim),
            initializer='zeros',
            trainable=True
        )
        
        self.rule_mf_indices = self._compute_rule_indices(n_features, self.n_mfs)
        super(FeatureGroupANFIS, self).build(input_shape)
    
    def _compute_rule_indices(self, n_features, n_mfs):
        indices = []
        for rule_idx in range(n_mfs ** n_features):
            rule_mfs = []
            temp = rule_idx
            for _ in range(n_features):
                rule_mfs.append(temp % n_mfs)
                temp //= n_mfs
            indices.append(rule_mfs)
        return tf.constant(indices, dtype=tf.int32)
    
    def call(self, inputs, return_firing_strengths=False):
        batch_size = tf.shape(inputs)[0]
        x_exp = tf.expand_dims(inputs, 2)
        c = tf.expand_dims(self.mf_centers, 0)
        s = tf.abs(self.mf_widths) + 0.1
        s = tf.expand_dims(s, 0)
        
        memberships = tf.exp(-tf.square(x_exp - c) / (2 * tf.square(s)))
        
        firing_strengths = tf.ones((batch_size, self.n_rules))
        for feat_idx in range(self.n_features):
            mf_indices = self.rule_mf_indices[:, feat_idx]
            feat_memberships = memberships[:, feat_idx, :]
            rule_memberships = tf.gather(feat_memberships, mf_indices, axis=1)
            firing_strengths = firing_strengths * rule_memberships
        
        firing_sum = tf.reduce_sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        firing_norm = firing_strengths / firing_sum
        
        x_exp2 = tf.expand_dims(inputs, 1)
        x_exp3 = tf.expand_dims(x_exp2, 3)
        p_exp = tf.expand_dims(self.consequent_p, 0)
        
        linear = tf.reduce_sum(x_exp3 * p_exp, axis=2)
        rule_outputs = linear + self.consequent_r
        
        firing_exp = tf.expand_dims(firing_norm, 2)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)
        
        if return_firing_strengths:
            return output, firing_norm
        return output

    def get_rule_descriptions(self, feature_names):
        mf_labels = ['LOW', 'HIGH'] if self.n_mfs == 2 else ['LOW', 'MEDIUM', 'HIGH']
        rules = []
        for rule_idx in range(self.n_rules):
            conditions = []
            for feat_idx in range(self.n_features):
                mf_idx = self.rule_mf_indices[rule_idx, feat_idx].numpy()
                conditions.append(f"{feature_names[feat_idx]} is {mf_labels[mf_idx]}")
            rule_str = f"Rule {rule_idx+1}: IF " + " AND ".join(conditions)
            rules.append(rule_str)
        return rules
    
    def get_config(self):
        config = super(FeatureGroupANFIS, self).get_config()
        config.update({
            'n_mfs': self.n_mfs,
            'output_dim': self.output_dim,
            'name_prefix': self.name_prefix
        })
        return config

# ===============================
# MODEL CREATION
# ===============================
def create_model(look_back, n_features, n_mfs=2, lstm_units=64, dropout=0.2, lr=0.001,
                 returns_centers=None, indic_centers=None):
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # Feature Group selection
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    returns_features = layers.Lambda(lambda x: x[:, :4], name='returns_slice')(last_step)
    indic_features = layers.Lambda(lambda x: x[:, 4:], name='indic_slice')(last_step)
    
    anfis_returns = FeatureGroupANFIS(
        n_mfs=n_mfs, output_dim=8, name_prefix='returns',
        initial_centers=returns_centers, name='anfis_returns'
    )(returns_features)
    
    anfis_indic = FeatureGroupANFIS(
        n_mfs=n_mfs, output_dim=4, name_prefix='indic',
        initial_centers=indic_centers, name='anfis_indic'
    )(indic_features)
    
    anfis_combined = layers.Concatenate(name='anfis_combined')([anfis_returns, anfis_indic])
    anfis_out = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_combined)
    
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm_1')(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units // 2), name='bilstm_2')(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    combined = layers.Concatenate(name='final_combine')([anfis_out, lstm_out])
    
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(4, name='output')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    return model

# ===============================
# DATA PREPARATION
# ===============================
def prepare_data(data, look_back=60, train_split=0.8):
    df = data.copy()
    
    # Features
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Targets (Next day)
    df['target_Close_ret'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target_Open_ret'] = df['Open'].shift(-1) / df['Close'] - 1
    df['target_High_ret'] = df['High'].shift(-1) / df['Close'] - 1
    df['target_Low_ret'] = df['Low'].shift(-1) / df['Close'] - 1
    
    # Next day prices (for reconstruction)
    df['next_Close'] = df['Close'].shift(-1)
    df['next_Open'] = df['Open'].shift(-1)
    df['next_High'] = df['High'].shift(-1)
    df['next_Low'] = df['Low'].shift(-1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    clip_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap', 
                 'target_Close_ret', 'target_Open_ret', 'target_High_ret', 'target_Low_ret']
    for col in clip_cols:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
            
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    target_cols = ['target_Close_ret', 'target_Open_ret', 'target_High_ret', 'target_Low_ret']
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    current_prices = df[['Close', 'Open', 'High', 'Low']].values
    next_day_prices = df[['next_Close', 'next_Open', 'next_High', 'next_Low']].values
    
    # Scaling
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    X, y, curr_px, next_px = [], [], [], []
    for i in range(look_back, len(features)):
        if i+1 <= len(features):
            X.append(scaled_feat[i-look_back+1 : i+1])
            y.append(scaled_tgt[i])
            curr_px.append(current_prices[i])
            next_px.append(next_day_prices[i])
            
    X = np.array(X)
    y = np.array(y)
    curr_px = np.array(curr_px)
    next_px = np.array(next_px)
    
    split = int(len(X) * train_split)
    
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'current_prices_train': curr_px[:split], 'current_prices_test': curr_px[split:],
        'actual_next_prices_train': next_px[:split], 'actual_next_prices_test': next_px[split:],
        'tgt_scaler': tgt_scaler,
        'train_indices': np.arange(split),
        'test_indices': np.arange(split, len(X))
    }

# ===============================
# METRICS & VISUALIZATION
# ===============================
def calculate_all_metrics(y_true, y_pred, y_prev, train_time, epochs, n_params):
    """
    Calculate comprehensive metrics for the report.
    Returns dictionary of metrics for each target + global metrics.
    """
    metrics = {}
    names = ['Close', 'Open', 'High', 'Low']
    
    # Per-variable metrics
    for i, n in enumerate(names):
        t = y_true[:, i]
        p = y_pred[:, i]
        
        mse = mean_squared_error(t, p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p)
        
        # MAPE (handle divide by zero)
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0.0
        
        # DA (Directional Accuracy)
        actual_dir = np.sign(t - y_prev[:, i])
        pred_dir = np.sign(p - y_prev[:, i])
        da = np.mean((actual_dir == pred_dir).astype(float)) * 100
        
        metrics[n] = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2),
            'DA': float(da)
        }
    
    # Global/Model metrics
    metrics['Model'] = {
        'Training_Time_Sec': float(train_time),
        'Convergence_Epochs': int(epochs),
        'Total_Parameters': int(n_params),
        'Avg_R2': float(np.mean([metrics[n]['R2'] for n in names])),
        'Avg_DA': float(np.mean([metrics[n]['DA'] for n in names]))
    }
    
    return metrics

def plot_results(history, y_true, y_pred, metrics, stock_name, output_dir):
    """
    Generate and save visualization plots.
    1. Loss Curve
    2. Actual vs Predicted (Time Series)
    3. Scatter Plot (Regression)
    """
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{stock_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_loss_curve.png'))
    plt.close()
    
    # 2. Actual vs Predicted (Close Price) - Zoomed in last 100 points if available
    plt.figure(figsize=(14, 7))
    limit = min(200, len(y_true))
    plt.plot(y_true[-limit:, 0], label='Actual Close', linewidth=2)
    plt.plot(y_pred[-limit:, 0], label='Predicted Close', linestyle='--', linewidth=2)
    plt.title(f'{stock_name} - Actual vs Predicted (Last {limit} Days)\nR²: {metrics["Close"]["R2"]:.4f}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{stock_name}_prediction_comparison.png'))
    plt.close()
    
    # 3. Scatter Plot (Close Price)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, color='blue', s=20)
    
    # Perfect fit line
    min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
    max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    
    plt.title(f'{stock_name} - Prediction Scatter Plot (Close)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, f'{stock_name}_scatter_plot.png'))
    plt.close()

# ===============================
# MAIN EXECUTOR
# ===============================
def run_final_model(stock_name, data_path, output_dir):
    print(f"\n🚀 STARTED processing {stock_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    data = pd.read_csv(data_path)
    d = prepare_data(data)
    
    # Initialize Centers
    print("   Computing initial centers (K-Means)...")
    train_last = d['X_train'][:, -1, :]
    
    # Returns (0-3)
    returns_centers = np.zeros((4, 2))
    for i in range(4):
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(train_last[:, i:i+1])
        returns_centers[i] = np.sort(km.cluster_centers_.flatten())
        
    # Indicators (4-5)
    indic_centers = np.zeros((2, 2))
    for i in range(2):
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(train_last[:, 4+i:4+i+1])
        indic_centers[i] = np.sort(km.cluster_centers_.flatten())
        
    # Create Model
    model = create_model(
        look_back=60, 
        n_features=d['X_train'].shape[-1], 
        n_mfs=2,
        returns_centers=returns_centers,
        indic_centers=indic_centers
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        LearningRateScheduler(lambda e: 0.001 * (0.95 ** e) if e > 20 else 0.001), # Decay
    ]
    
    # Train
    print("   Training model...")
    start_time = time.time()
    history = model.fit(
        d['X_train'], d['y_train'],
        validation_data=(d['X_test'], d['y_test']),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    epochs_trained = len(history.history['loss'])
    
    # Predict
    print("   Generating predictions...")
    pred_scaled = model.predict(d['X_test'], verbose=0)
    pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
    
    # Reconstruct Prices
    current_close_test = d['current_prices_test'][:, 0:1] # Close price
    pred_px = current_close_test * (1 + pred_ret)
    actual_px = d['actual_next_prices_test']
    current_px = d['current_prices_test']
    
    # Calculate Metrics
    results_metrics = calculate_all_metrics(
        actual_px, pred_px, current_px,
        train_time, epochs_trained, model.count_params()
    )
    
    # Print Summary
    print("\n" + "="*50)
    print(f"📊 RESULTS: {stock_name}")
    print("="*50)
    for k, v in results_metrics['Close'].items():
        print(f"   {k}: {v:.4f}")
    print(f"   Time: {train_time:.2f}s")
    
    # Save Model
    model_save_path = os.path.join(output_dir, f'{stock_name}_final_model.keras')
    model.save(model_save_path)
    print(f"💾 Model saved to {model_save_path}")
    
    # Visualize
    plot_results(history.history, actual_px, pred_px, results_metrics, stock_name, output_dir)
    print("🖼️  Visualizations saved.")
    
    # Save Metrics JSON
    json_path = os.path.join(output_dir, f'{stock_name}_final_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results_metrics, f, indent=4)
        
    print(f"📝 Metrics saved to {json_path}")
    print("✅ DONE.\n")

if __name__ == '__main__':
    # Configuration
    BASE_DIR = '/Users/bao/Documents/tsa_paper_1'
    OUTPUT_DIR = os.path.join(BASE_DIR, 'final_report_outputs')
    
    # Add stocks here
    STOCKS = {
        'IHSG': os.path.join(BASE_DIR, 'IHSG_2007_2024.csv'),
        'AMZN': os.path.join(BASE_DIR, 'AMZN.csv'),
        'JPM': os.path.join(BASE_DIR, 'JPM.csv'),
        'TSLA': os.path.join(BASE_DIR, 'TSLA.csv'),
        'PingAn_Bank': os.path.join(BASE_DIR, 'PingAn_Bank_2010_2023.csv'),
        'Sinopharm': os.path.join(BASE_DIR, 'Sinopharm_2010_2023.csv'),
        'ChinaSouth_Publishing': os.path.join(BASE_DIR, 'ChinaSouth_Publishing_2010_2023.csv'),
    }
    
    for stock, path in STOCKS.items():
        if os.path.exists(path):
            try:
                run_final_model(stock, path, OUTPUT_DIR)
            except Exception as e:
                print(f"❌ Error training {stock}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ File not found: {path}")
