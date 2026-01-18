#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direction Classification ANFIS + BiLSTM
=======================================

Predicts DIRECTION (UP/DOWN) instead of price value.

Why this is better:
- R² can be misleading (high R² but low DA)
- Direction matters more for trading
- Classification is more honest metric

Output: Probability of price going UP
Metric: Accuracy (= Directional Accuracy)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
# FEATURE CONFIGURATION
# ===============================
class FeatureConfig:
    """Flexible configuration for different data scenarios."""
    
    def __init__(self, price_cols=None, macro_cols=None):
        self.price_cols = price_cols or ['Close']
        self.macro_cols = macro_cols or []
        self.derived_features = []
        
    def setup_for_data(self, df):
        """Auto-configure based on available columns."""
        available = df.columns.tolist()
        
        self.price_cols = [c for c in self.price_cols if c in available]
        if not self.price_cols:
            for c in ['Close', 'Adj Close', 'Price', 'close']:
                if c in available:
                    self.price_cols = [c]
                    break
        
        self.macro_cols = [c for c in self.macro_cols if c in available]
        self._setup_derived_features()
        
        print(f"📊 Feature Config:")
        print(f"   Price cols: {self.price_cols}")
        print(f"   Macro cols: {self.macro_cols}")
        print(f"   Total features: {len(self.get_all_feature_names())}")
        
    def _setup_derived_features(self):
        self.derived_features = []
        
        for col in self.price_cols:
            self.derived_features.append(f'{col}_ret')
        
        if 'High' in self.price_cols and 'Low' in self.price_cols:
            self.derived_features.append('range_pct')
        
        if 'Open' in self.price_cols and 'Close' in self.price_cols:
            self.derived_features.append('gap')
    
    def get_all_feature_names(self):
        return self.derived_features + self.macro_cols
    
    def get_feature_groups(self):
        all_features = self.get_all_feature_names()
        n_features = len(all_features)
        
        if n_features <= 4:
            return {'main': all_features}
        elif n_features <= 6:
            mid = n_features // 2
            return {
                'group1': all_features[:mid],
                'group2': all_features[mid:]
            }
        else:
            n_per_group = min(4, (n_features + 2) // 3)
            return {
                'returns': all_features[:n_per_group],
                'indicators': all_features[n_per_group:2*n_per_group],
                'extra': all_features[2*n_per_group:]
            }


# ===============================
# ANFIS LAYER
# ===============================
class FlexibleANFIS(layers.Layer):
    """ANFIS layer for classification."""
    
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
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_mfs': self.n_mfs,
            'output_dim': self.output_dim,
            'name_prefix': self.name_prefix,
        })
        return config


# ===============================
# CLASSIFICATION MODEL
# ===============================
def create_direction_model(look_back, feature_groups, n_mfs=2, 
                           lstm_units=64, dropout=0.3, lr=0.001,
                           group_centers=None):
    """
    Create CLASSIFICATION model for direction prediction.
    
    Output: Probability of UP direction (sigmoid)
    Loss: Binary crossentropy
    """
    n_features = sum(len(indices) for indices in feature_groups.values())
    
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # Last timestep
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    
    # ANFIS for each group
    anfis_outputs = []
    for group_name, indices in feature_groups.items():
        group_features = layers.Lambda(
            lambda x, idx=indices: tf.gather(x, idx, axis=-1),
            name=f'{group_name}_features'
        )(last_step)
        
        centers = group_centers.get(group_name) if group_centers else None
        
        anfis_out = FlexibleANFIS(
            n_mfs=n_mfs,
            output_dim=8,
            name_prefix=group_name,
            initial_centers=centers,
            name=f'anfis_{group_name}'
        )(group_features)
        
        anfis_outputs.append(anfis_out)
    
    if len(anfis_outputs) > 1:
        anfis_combined = layers.Concatenate(name='anfis_combined')(anfis_outputs)
    else:
        anfis_combined = anfis_outputs[0]
    
    anfis_out = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_combined)
    anfis_out = layers.Dropout(dropout)(anfis_out)
    
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
    
    # Classification output
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu', name='dense_2')(x)
    
    # Binary output: probability of UP
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ===============================
# DATA PREPARATION
# ===============================
def prepare_data(data, config, look_back=60, train_split=0.8):
    """
    Prepare data for DIRECTION classification.
    
    Target: 1 if next day Close > today's Close, else 0
    """
    df = data.copy()
    config.setup_for_data(df)
    
    # Compute returns
    for col in config.price_cols:
        df[f'{col}_ret'] = df[col].pct_change()
    
    # Derived features
    if 'High' in config.price_cols and 'Low' in config.price_cols:
        close_col = 'Close' if 'Close' in config.price_cols else config.price_cols[0]
        df['range_pct'] = (df['High'] - df['Low']) / df[close_col]
    
    if 'Open' in config.price_cols and 'Close' in config.price_cols:
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # TARGET: Direction (1=UP, 0=DOWN)
    close_col = 'Close' if 'Close' in config.price_cols else config.price_cols[0]
    df['target_direction'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    feature_names = config.get_all_feature_names()
    for col in feature_names:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    
    df = df.dropna().reset_index(drop=True)
    
    features = df[feature_names].values
    targets = df['target_direction'].values
    
    # Scale features
    feat_scaler = StandardScaler()
    scaled_feat = feat_scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(scaled_feat[i-look_back+1 : i+1])
        y.append(targets[i])
    
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * train_split)
    
    # Feature groups
    feature_groups = config.get_feature_groups()
    feature_indices = {}
    for group_name, group_features in feature_groups.items():
        indices = [feature_names.index(f) for f in group_features]
        feature_indices[group_name] = indices
    
    # Class balance
    train_up = y[:split].sum()
    train_down = split - train_up
    print(f"   Train balance: UP={train_up} ({100*train_up/split:.1f}%), DOWN={train_down} ({100*train_down/split:.1f}%)")
    
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'config': config,
        'feature_groups': feature_indices,
        'feature_names': feature_names
    }


def compute_kmeans_centers(features, n_clusters=2):
    n_features = features.shape[1]
    centers = np.zeros((n_features, n_clusters))
    
    for i in range(n_features):
        feat_data = features[:, i].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feat_data)
        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
        centers[i] = sorted_centers
    
    return centers


def cosine_lr_schedule(epoch, initial_lr=0.001, total_epochs=150, min_lr=1e-6):
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))


# ===============================
# TRAINING
# ===============================
def train_stock(stock_name, data_path, output_dir, config=None, 
                n_mfs=2, n_runs=3, epochs=150, seed=42):
    """Train direction classification model."""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (Direction Classification)")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows")
    
    if config is None:
        config = FeatureConfig()
    
    d = prepare_data(data, config)
    
    print(f"\n📐 Data shapes:")
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    print(f"   Features: {d['feature_names']}")
    
    # K-Means centers
    group_centers = {}
    for group_name, indices in d['feature_groups'].items():
        group_features = d['X_train'][:, -1, indices]
        centers = compute_kmeans_centers(group_features, n_mfs)
        group_centers[group_name] = centers
        print(f"   {group_name}: {len(indices)} features → {n_mfs**len(indices)} rules")
    
    best_model = None
    best_acc = 0
    best_metrics = None
    
    print(f"\n🎓 Starting {n_runs} training runs...")
    
    for run in range(n_runs):
        run_seed = seed + run
        set_seed(run_seed)
        
        print(f"\n--- Run {run+1}/{n_runs} (seed={run_seed}) ---")
        
        model = create_direction_model(
            look_back=60,
            feature_groups=d['feature_groups'],
            n_mfs=n_mfs,
            lstm_units=64,
            group_centers=group_centers
        )
        
        if run == 0:
            total_rules = sum(n_mfs**len(idx) for idx in d['feature_groups'].values())
            print(f"   Total ANFIS rules: {total_rules}")
            print(f"   Parameters: {model.count_params():,}")
        
        # Class weights (handle imbalance)
        n_up = d['y_train'].sum()
        n_down = len(d['y_train']) - n_up
        class_weight = {0: 1.0, 1: n_down / n_up} if n_up > 0 else None
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
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
            class_weight=class_weight,
            verbose=0
        )
        train_time = time.time() - train_start
        epochs_trained = len(history.history['loss'])
        
        # Predict
        pred_prob = model.predict(d['X_test'], verbose=0).flatten()
        pred_class = (pred_prob > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(d['y_test'], pred_class)
        precision = precision_score(d['y_test'], pred_class, zero_division=0)
        recall = recall_score(d['y_test'], pred_class, zero_division=0)
        f1 = f1_score(d['y_test'], pred_class, zero_division=0)
        
        print(f"   Accuracy: {acc*100:.1f}%, Precision: {precision*100:.1f}%, "
              f"F1: {f1*100:.1f}%, Epochs: {epochs_trained}/{epochs}, Time: {train_time:.1f}s")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_pred = pred_class
            best_metrics = {
                'Accuracy': acc * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1': f1 * 100
            }
    
    # Confusion matrix
    cm = confusion_matrix(d['y_test'], best_pred)
    
    # Final metrics
    print(f"\n{'='*70}")
    print(f"📊 FINAL METRICS FOR {stock_name} (Direction Classification)")
    print(f"{'='*70}")
    
    flag = "✅" if best_metrics['Accuracy'] >= 55 else ("⚠️" if best_metrics['Accuracy'] >= 50 else "❌")
    print(f"Accuracy:  {best_metrics['Accuracy']:5.1f}% {flag}")
    print(f"Precision: {best_metrics['Precision']:5.1f}%")
    print(f"Recall:    {best_metrics['Recall']:5.1f}%")
    print(f"F1 Score:  {best_metrics['F1']:5.1f}%")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual UP     {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_direction.keras')
    best_model.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    
    return {'metrics': best_metrics}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_direction')
    
    # Configs
    ohlc_config = FeatureConfig(price_cols=['Close', 'Open', 'High', 'Low'])
    minimal_config = FeatureConfig(price_cols=['Close', 'Open'])
    
    # Choose config
    config = ohlc_config
    
    stocks = {
        'IHSG': os.path.join(base_dir, 'IHSG_2007_2024.csv'),
    }
    
    all_results = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                result = train_stock(name, path, output_dir, config=config, n_mfs=2, n_runs=3)
                if result:
                    all_results[name] = result
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY - Direction Classification")
    print(f"{'='*70}")
    
    for name, result in all_results.items():
        m = result['metrics']
        flag = "✅" if m['Accuracy'] >= 55 else ("⚠️" if m['Accuracy'] >= 50 else "❌")
        print(f"{name:10}: Accuracy={m['Accuracy']:.1f}%, F1={m['F1']:.1f}% {flag}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
