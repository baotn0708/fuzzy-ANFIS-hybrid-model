#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible Feature-Group ANFIS + BiLSTM
=====================================

FLEXIBLE CONFIGURATION:
- Auto-detects available columns (Close, Open, High, Low, macro vars)
- Configurable feature groups 
- Works with minimal data (Close only) or rich data (OHLC + macro)

Usage:
    # Minimal config (Close/Open only)
    config = FeatureConfig(price_cols=['Close', 'Open'])
    
    # Full OHLC
    config = FeatureConfig(price_cols=['Close', 'Open', 'High', 'Low'])
    
    # With macro variables
    config = FeatureConfig(
        price_cols=['Close', 'Open', 'High', 'Low'],
        macro_cols=['GDP', 'CPI', 'Interest_Rate']
    )
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import math
import json
import time
import warnings
warnings.filterwarnings('ignore')

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
# FLEXIBLE FEATURE CONFIGURATION
# ===============================
class FeatureConfig:
    """
    Flexible configuration for different data scenarios.
    
    Examples:
        # Minimal: Close only
        config = FeatureConfig(price_cols=['Close'])
        
        # Standard: OHLC
        config = FeatureConfig(price_cols=['Close', 'Open', 'High', 'Low'])
        
        # With macro variables
        config = FeatureConfig(
            price_cols=['Close', 'Open'],
            macro_cols=['GDP_growth', 'Inflation', 'Interest_Rate']
        )
    """
    
    def __init__(self, price_cols=None, macro_cols=None, target_cols=None):
        # Default: use whatever price columns are available
        self.price_cols = price_cols or ['Close']
        self.macro_cols = macro_cols or []
        
        # Target: predict returns for price columns (default: Close only)
        self.target_cols = target_cols or ['Close']
        
        # Derived features (auto-computed from price columns)
        self.derived_features = []
        
    def setup_for_data(self, df):
        """Auto-configure based on available columns in data."""
        available = df.columns.tolist()
        
        # Filter to available price columns
        self.price_cols = [c for c in self.price_cols if c in available]
        if not self.price_cols:
            # Fallback: find any price-like column
            for c in ['Close', 'Adj Close', 'Price', 'close', 'CLOSE']:
                if c in available:
                    self.price_cols = [c]
                    break
        
        # Filter macro columns
        self.macro_cols = [c for c in self.macro_cols if c in available]
        
        # Ensure targets are in price_cols
        self.target_cols = [c for c in self.target_cols if c in self.price_cols]
        if not self.target_cols:
            self.target_cols = self.price_cols[:1]  # At least Close
        
        # Setup derived features
        self._setup_derived_features()
        
        print(f"📊 Feature Config:")
        print(f"   Price cols: {self.price_cols}")
        print(f"   Macro cols: {self.macro_cols}")
        print(f"   Target cols: {self.target_cols}")
        print(f"   Total features: {len(self.get_all_feature_names())}")
        
    def _setup_derived_features(self):
        """Setup derived features based on available price columns."""
        self.derived_features = []
        
        # Returns for each price column
        for col in self.price_cols:
            self.derived_features.append(f'{col}_ret')
        
        # Range percent (if High/Low available)
        if 'High' in self.price_cols and 'Low' in self.price_cols:
            self.derived_features.append('range_pct')
        
        # Gap (if Open and Close available)
        if 'Open' in self.price_cols and 'Close' in self.price_cols:
            self.derived_features.append('gap')
    
    def get_all_feature_names(self):
        """Get all feature names (derived + macro)."""
        return self.derived_features + self.macro_cols
    
    def get_feature_groups(self):
        """
        Split features into groups for ANFIS.
        Returns dict: {'group_name': [feature_names]}
        
        Strategy: Keep groups small (≤4 features) to avoid rule explosion
        """
        all_features = self.get_all_feature_names()
        n_features = len(all_features)
        
        if n_features <= 4:
            # Single group
            return {'main': all_features}
        elif n_features <= 6:
            # Split into 2 groups
            mid = n_features // 2
            return {
                'group1': all_features[:mid],
                'group2': all_features[mid:]
            }
        else:
            # Split into 3 groups (max 4 each to avoid 2^5+ rules)
            n_per_group = min(4, (n_features + 2) // 3)
            return {
                'returns': all_features[:n_per_group],
                'indicators': all_features[n_per_group:2*n_per_group],
                'extra': all_features[2*n_per_group:]
            }
    
    def get_target_names(self):
        """Get target column names (returns)."""
        return [f'{col}_ret' for col in self.target_cols]


# ===============================
# ANFIS LAYER
# ===============================
class FlexibleANFIS(layers.Layer):
    """
    Flexible ANFIS layer that adapts to any number of input features.
    """
    
    def __init__(self, n_mfs=2, output_dim=8, name_prefix='anfis', 
                 initial_centers=None, **kwargs):
        super(FlexibleANFIS, self).__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.initial_centers = initial_centers
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        self.n_rules = self.n_mfs ** n_features
        
        # Initialize MF centers
        if self.initial_centers is not None and self.initial_centers.shape[0] == n_features:
            center_init = tf.constant_initializer(self.initial_centers)
        else:
            center_init = tf.keras.initializers.RandomUniform(-1.0, 1.0)
        
        self.mf_centers = self.add_weight(
            name=f'{self.name_prefix}_centers',
            shape=(n_features, self.n_mfs),
            initializer=center_init,
            trainable=True
        )
        
        self.mf_widths = self.add_weight(
            name=f'{self.name_prefix}_widths',
            shape=(n_features, self.n_mfs),
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )
        
        # TSK consequents
        self.consequent_p = self.add_weight(
            name=f'{self.name_prefix}_p',
            shape=(self.n_rules, n_features, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.consequent_r = self.add_weight(
            name=f'{self.name_prefix}_r',
            shape=(self.n_rules, self.output_dim),
            initializer='zeros',
            trainable=True
        )
        
        self.rule_mf_indices = self._compute_rule_indices(n_features, self.n_mfs)
        super(FlexibleANFIS, self).build(input_shape)
    
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
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Fuzzification
        x_exp = tf.expand_dims(inputs, 2)
        c = tf.expand_dims(self.mf_centers, 0)
        s = tf.abs(self.mf_widths) + 0.1
        s = tf.expand_dims(s, 0)
        
        memberships = tf.exp(-tf.square(x_exp - c) / (2 * tf.square(s)))
        
        # Rule firing
        firing_strengths = tf.ones((batch_size, self.n_rules))
        for feat_idx in range(self.n_features):
            mf_indices = self.rule_mf_indices[:, feat_idx]
            feat_memberships = memberships[:, feat_idx, :]
            rule_memberships = tf.gather(feat_memberships, mf_indices, axis=1)
            firing_strengths = firing_strengths * rule_memberships
        
        # Normalize
        firing_sum = tf.reduce_sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        firing_norm = firing_strengths / firing_sum
        
        # TSK output
        x_exp2 = tf.expand_dims(inputs, 1)
        x_exp3 = tf.expand_dims(x_exp2, 3)
        p_exp = tf.expand_dims(self.consequent_p, 0)
        
        linear = tf.reduce_sum(x_exp3 * p_exp, axis=2)
        rule_outputs = linear + self.consequent_r
        
        # Defuzzification
        firing_exp = tf.expand_dims(firing_norm, 2)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)
        
        return output
    
    def get_rule_descriptions(self, feature_names):
        """Get human-readable rule descriptions."""
        mf_labels = ['LOW', 'HIGH'] if self.n_mfs == 2 else ['LOW', 'MED', 'HIGH']
        rules = []
        for rule_idx in range(self.n_rules):
            conditions = []
            for feat_idx in range(self.n_features):
                mf_idx = self.rule_mf_indices[rule_idx, feat_idx].numpy()
                conditions.append(f"{feature_names[feat_idx]} is {mf_labels[mf_idx]}")
            rules.append(f"Rule {rule_idx+1}: IF " + " AND ".join(conditions))
        return rules
    
    def get_config(self):
        """Required for model serialization."""
        config = super().get_config()
        config.update({
            'n_mfs': self.n_mfs,
            'output_dim': self.output_dim,
            'name_prefix': self.name_prefix,
        })
        return config



# ===============================
# MODEL BUILDER
# ===============================
def create_flexible_model(look_back, feature_groups, n_targets, n_mfs=2, 
                          lstm_units=64, dropout=0.2, lr=0.001,
                          group_centers=None):
    """
    Create model that adapts to any feature configuration.
    
    Args:
        feature_groups: dict of {group_name: [feature_indices]}
        n_targets: number of target outputs
        group_centers: dict of {group_name: initial_centers}
    """
    # Calculate total features
    n_features = sum(len(indices) for indices in feature_groups.values())
    
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # Last timestep features
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    
    # ANFIS for each feature group
    anfis_outputs = []
    for group_name, indices in feature_groups.items():
        n_group_features = len(indices)
        
        # Extract group features
        group_features = layers.Lambda(
            lambda x, idx=indices: tf.gather(x, idx, axis=-1),
            name=f'{group_name}_features'
        )(last_step)
        
        # Get initial centers if available
        centers = group_centers.get(group_name) if group_centers else None
        
        # ANFIS layer
        anfis_out = FlexibleANFIS(
            n_mfs=n_mfs,
            output_dim=8,
            name_prefix=group_name,
            initial_centers=centers,
            name=f'anfis_{group_name}'
        )(group_features)
        
        anfis_outputs.append(anfis_out)
    
    # Combine ANFIS outputs
    if len(anfis_outputs) > 1:
        anfis_combined = layers.Concatenate(name='anfis_combined')(anfis_outputs)
    else:
        anfis_combined = anfis_outputs[0]
    
    anfis_out = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_combined)
    
    # BiLSTM
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name='bilstm_1'
    )(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units // 2),
        name='bilstm_2'
    )(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    # Combine
    combined = layers.Concatenate(name='final_combine')([anfis_out, lstm_out])
    
    # Output
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(n_targets, name='output')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model


# ===============================
# DATA PREPARATION
# ===============================
def prepare_data(data, config, look_back=60, train_split=0.8):
    """
    Prepare data based on configuration.
    
    Returns dict with X_train, X_test, y_train, y_test, etc.
    """
    df = data.copy()
    
    # Setup config for this data
    config.setup_for_data(df)
    
    # Compute returns for price columns
    for col in config.price_cols:
        df[f'{col}_ret'] = df[col].pct_change()
    
    # Derived features
    if 'High' in config.price_cols and 'Low' in config.price_cols:
        close_col = 'Close' if 'Close' in config.price_cols else config.price_cols[0]
        df['range_pct'] = (df['High'] - df['Low']) / df[close_col]
    
    if 'Open' in config.price_cols and 'Close' in config.price_cols:
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Target returns (next day)
    target_cols = []
    for col in config.target_cols:
        target_name = f'target_{col}_ret'
        df[target_name] = df[col].shift(-1) / df[col] - 1
        target_cols.append(target_name)
    
    # Next day prices for evaluation
    for col in config.target_cols:
        df[f'next_{col}'] = df[col].shift(-1)
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    feature_names = config.get_all_feature_names()
    for col in feature_names:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    
    df = df.dropna().reset_index(drop=True)
    
    # Extract arrays
    features = df[feature_names].values
    targets = df[target_cols].values
    current_prices = df[config.target_cols].values
    next_prices = df[[f'next_{c}' for c in config.target_cols]].values
    
    # Scale
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    # Create sequences
    X, y, curr_px, next_px = [], [], [], []
    for i in range(look_back, len(features)):
        X.append(scaled_feat[i-look_back+1 : i+1])
        y.append(scaled_tgt[i])
        curr_px.append(current_prices[i])
        next_px.append(next_prices[i])
    
    X, y, curr_px, next_px = np.array(X), np.array(y), np.array(curr_px), np.array(next_px)
    
    split = int(len(X) * train_split)
    
    # Compute feature group indices
    feature_groups = config.get_feature_groups()
    feature_indices = {}
    for group_name, group_features in feature_groups.items():
        indices = [feature_names.index(f) for f in group_features]
        feature_indices[group_name] = indices
    
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'current_prices': curr_px[split:],
        'actual_next_prices': next_px[split:],
        'tgt_scaler': tgt_scaler,
        'config': config,
        'feature_groups': feature_indices,
        'feature_names': feature_names,
        'target_names': config.target_cols
    }


# ===============================
# METRICS
# ===============================
def calc_metrics(y_true, y_pred, y_prev=None, names=None):
    """Calculate metrics for each target."""
    names = names or [f'Target_{i}' for i in range(y_true.shape[1])]
    results = {}
    
    for i, n in enumerate(names):
        t = y_true[:, i] if y_true.ndim > 1 else y_true
        p = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        
        da = 0
        if y_prev is not None:
            prev = y_prev[:, i] if y_prev.ndim > 1 else y_prev
            actual_dir = np.sign(t - prev)
            pred_dir = np.sign(p - prev)
            da = np.mean((actual_dir == pred_dir).astype(float)) * 100
        
        results[n] = {
            'RMSE': math.sqrt(mean_squared_error(t, p)),
            'MAE': mean_absolute_error(t, p),
            'MAPE': mape,
            'R2': r2_score(t, p),
            'DA': da
        }
    
    return results


def cosine_lr_schedule(epoch, initial_lr=0.001, total_epochs=150, min_lr=1e-6):
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))


def compute_kmeans_centers(features, n_clusters=2):
    """Compute K-Means centers for each feature."""
    n_features = features.shape[1]
    centers = np.zeros((n_features, n_clusters))
    
    for i in range(n_features):
        feat_data = features[:, i].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feat_data)
        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
        centers[i] = sorted_centers
    
    return centers


# ===============================
# TRAINING
# ===============================
def train_stock(stock_name, data_path, output_dir, config=None, 
                n_mfs=2, n_runs=3, epochs=150, seed=42):
    """
    Train model with flexible configuration.
    """
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name}")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows, columns: {list(data.columns)}")
    
    # Use default config if not provided
    if config is None:
        config = FeatureConfig()
    
    # Prepare data
    d = prepare_data(data, config)
    
    print(f"\n📐 Data shapes:")
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    print(f"   Features: {d['feature_names']}")
    print(f"   Targets: {d['target_names']}")
    
    # Compute K-Means centers for each group
    group_centers = {}
    for group_name, indices in d['feature_groups'].items():
        group_features = d['X_train'][:, -1, indices]
        centers = compute_kmeans_centers(group_features, n_mfs)
        group_centers[group_name] = centers
        print(f"   {group_name}: {len(indices)} features → {n_mfs**len(indices)} rules")
    
    best_model = None
    best_r2 = -np.inf
    best_metrics = None
    
    print(f"\n🎓 Starting {n_runs} training runs...")
    
    for run in range(n_runs):
        run_seed = seed + run
        set_seed(run_seed)
        
        print(f"\n--- Run {run+1}/{n_runs} (seed={run_seed}) ---")
        
        model = create_flexible_model(
            look_back=60,
            feature_groups=d['feature_groups'],
            n_targets=len(d['target_names']),
            n_mfs=n_mfs,
            lstm_units=64,
            group_centers=group_centers
        )
        
        if run == 0:
            total_rules = sum(n_mfs**len(idx) for idx in d['feature_groups'].values())
            print(f"   Total ANFIS rules: {total_rules}")
            print(f"   Parameters: {model.count_params():,}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            LearningRateScheduler(lambda e: cosine_lr_schedule(e, 0.001, epochs)),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        ]
        
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
        
        # Predict
        pred_scaled = model.predict(d['X_test'], verbose=0)
        pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
        
        # Convert to prices
        current_close = d['current_prices'][:, 0:1]
        pred_px = current_close * (1 + pred_ret)
        actual_px = d['actual_next_prices'][:len(pred_px)]
        current_px = d['current_prices'][:len(pred_px)]
        
        metrics = calc_metrics(actual_px, pred_px, current_px, d['target_names'])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        
        first_target = d['target_names'][0]
        print(f"   {first_target} R²: {metrics[first_target]['R2']:.4f}, "
              f"DA: {metrics[first_target]['DA']:.1f}%, "
              f"Epochs: {epochs_trained}/{epochs}, Time: {train_time:.1f}s")
        
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = model
            best_metrics = metrics
            best_pred_px = pred_px
    
    # Final metrics
    print(f"\n{'='*70}")
    print(f"📊 FINAL METRICS FOR {stock_name}")
    print(f"{'='*70}")
    
    for n, m in best_metrics.items():
        flag = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:10} - RMSE: {m['RMSE']:8.2f} | MAPE: {m['MAPE']:5.2f}% | "
              f"R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {flag}")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_model.keras')
    best_model.save(model_path)
    
    return {'metrics': best_metrics, 'config': config}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_flexible')
    
    # === CONFIGURATION EXAMPLES ===
    
    # Standard OHLC config
    ohlc_config = FeatureConfig(
        price_cols=['Close', 'Open', 'High', 'Low'],
        target_cols=['Close']
    )
    
    # Minimal config (Close/Open only)
    minimal_config = FeatureConfig(
        price_cols=['Close', 'Open'],
        target_cols=['Close']
    )
    
    # With macro variables (if available)
    macro_config = FeatureConfig(
        price_cols=['Close', 'Open', 'High', 'Low'],
        macro_cols=['GDP', 'CPI', 'Interest'],  # Add your macro columns
        target_cols=['Close']
    )
    
    # === CHOOSE CONFIG ===
    config = minimal_config  # Change this as needed
    
    stocks = {
        'IHSG': os.path.join(base_dir, 'IHSG_2007_2024.csv'),
    }
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                train_stock(name, path, output_dir, config=config, n_mfs=2, n_runs=1)
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
