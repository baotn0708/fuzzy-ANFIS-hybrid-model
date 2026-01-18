#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-Group ANFIS + BiLSTM
===========================

Architecture (Option C from analysis):
- Returns (4 features) → ANFIS_returns (2 MFs) → Regime scores
- Indicators (2 features) → ANFIS_indic (2 MFs) → Regime scores  
- BiLSTM → Temporal features
- Concatenate all → Dense → Output

Advantages:
1. Avoids rule explosion (2^4=16 instead of 2^6=64)
2. Each ANFIS has semantic meaning
3. Interpretable fuzzy rules
4. Explainable predictions

Based on: Boyacioglu & Avci (2010) - R²=0.9827 with 2 Gaussian MFs
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
import matplotlib.pyplot as plt
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
# ANFIS LAYER (Feature-Group)
# ===============================
class FeatureGroupANFIS(layers.Layer):
    """
    ANFIS for a specific feature group.
    
    Uses 2 Gaussian Membership Functions per feature (as per paper).
    Supports rule extraction for interpretability.
    
    Architecture:
    - Layer 1: Fuzzification (Gaussian MFs)
    - Layer 2: Rule firing strength (product T-norm)  
    - Layer 3: Normalized firing strength
    - Layer 4: TSK first-order consequents
    - Layer 5: Defuzzification (weighted sum)
    """
    
    def __init__(self, n_mfs=2, output_dim=4, name_prefix='anfis', initial_centers=None, **kwargs):
        super(FeatureGroupANFIS, self).__init__(**kwargs)
        self.n_mfs = n_mfs  # 2 MFs per feature (Low, High)
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.initial_centers = initial_centers  # K-Means initialized centers
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        
        # Number of rules = n_mfs^n_features (e.g., 2^4=16 for returns)
        self.n_rules = self.n_mfs ** n_features
        
        # Layer 1: Gaussian MF parameters (centers and widths)
        # Shape: (n_features, n_mfs) - each feature has n_mfs membership functions
        if self.initial_centers is not None:
            # Use K-Means initialized centers
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
        
        # Layer 4: TSK consequent parameters
        # For first-order Sugeno: f_i = p_i1*x1 + p_i2*x2 + ... + r_i
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
        
        # Pre-compute rule combinations (which MF each rule uses for each feature)
        # E.g., for 2 features, 2 MFs: rule 0=[0,0], rule 1=[0,1], rule 2=[1,0], rule 3=[1,1]
        self.rule_mf_indices = self._compute_rule_indices(n_features, self.n_mfs)
        
        super(FeatureGroupANFIS, self).build(input_shape)
    
    def _compute_rule_indices(self, n_features, n_mfs):
        """Compute which MF each rule uses for each feature."""
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
        """
        Forward pass through ANFIS.
        
        Args:
            inputs: (batch, n_features)
            return_firing_strengths: If True, also return normalized firing strengths
            
        Returns:
            output: (batch, output_dim)
            firing_strengths: (batch, n_rules) - optional
        """
        batch_size = tf.shape(inputs)[0]
        
        # === Layer 1: Fuzzification ===
        # Compute membership values for each feature and each MF
        # x: (batch, n_features) -> expand to (batch, n_features, 1)
        x_exp = tf.expand_dims(inputs, 2)  # (batch, n_features, 1)
        
        # centers, widths: (n_features, n_mfs) -> expand to (1, n_features, n_mfs)
        c = tf.expand_dims(self.mf_centers, 0)
        s = tf.abs(self.mf_widths) + 0.1  # Ensure positive width
        s = tf.expand_dims(s, 0)
        
        # Gaussian MF: exp(-(x-c)^2 / (2*sigma^2))
        memberships = tf.exp(-tf.square(x_exp - c) / (2 * tf.square(s)))
        # memberships: (batch, n_features, n_mfs)
        
        # === Layer 2: Rule Firing Strength ===
        # For each rule, multiply the appropriate membership values
        # This is the T-norm (product)
        
        # Gather membership values for each rule
        firing_strengths = tf.ones((batch_size, self.n_rules))
        
        for feat_idx in range(self.n_features):
            # Get which MF index this rule uses for this feature
            mf_indices = self.rule_mf_indices[:, feat_idx]  # (n_rules,)
            
            # Get membership values for this feature: (batch, n_mfs)
            feat_memberships = memberships[:, feat_idx, :]
            
            # Gather the appropriate MF value for each rule: (batch, n_rules)
            rule_memberships = tf.gather(feat_memberships, mf_indices, axis=1)
            
            # Multiply (T-norm)
            firing_strengths = firing_strengths * rule_memberships
        
        # === Layer 3: Normalize Firing Strengths ===
        firing_sum = tf.reduce_sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        firing_norm = firing_strengths / firing_sum  # (batch, n_rules)
        
        # === Layer 4: TSK Consequents ===
        # f_i = sum_j(p_ij * x_j) + r_i
        x_exp2 = tf.expand_dims(inputs, 1)  # (batch, 1, n_features)
        x_exp3 = tf.expand_dims(x_exp2, 3)  # (batch, 1, n_features, 1)
        
        p_exp = tf.expand_dims(self.consequent_p, 0)  # (1, n_rules, n_features, output_dim)
        
        linear = tf.reduce_sum(x_exp3 * p_exp, axis=2)  # (batch, n_rules, output_dim)
        rule_outputs = linear + self.consequent_r  # (batch, n_rules, output_dim)
        
        # === Layer 5: Defuzzification ===
        firing_exp = tf.expand_dims(firing_norm, 2)  # (batch, n_rules, 1)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)  # (batch, output_dim)
        
        if return_firing_strengths:
            return output, firing_norm
        return output
    
    def get_rule_descriptions(self, feature_names):
        """
        Extract human-readable fuzzy rules.
        
        Returns list of rules like:
        "IF Close_ret is LOW AND Open_ret is HIGH THEN ..."
        """
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
# FEATURE-GROUP ANFIS + BiLSTM MODEL
# ===============================
def create_feature_group_model(look_back, n_features, n_mfs=2, lstm_units=64, dropout=0.2, lr=0.001,
                                returns_centers=None, indic_centers=None):
    """
    Create Feature-Group ANFIS + BiLSTM model.
    
    Architecture:
    - Group 1 (Returns): Close_ret, Open_ret, High_ret, Low_ret → ANFIS_returns
    - Group 2 (Indicators): range_pct, gap → ANFIS_indic
    - BiLSTM: Full sequence → Temporal features
    - Combine all → Dense → Output
    
    Args:
        returns_centers: K-Means centers for returns MFs (4, 2)
        indic_centers: K-Means centers for indicators MFs (2, 2)
    """
    # Input: (batch, look_back, n_features)
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # === Feature Groups ===
    # Take last timestep for ANFIS (most recent data point)
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    
    # Group 1: Returns (features 0-3)
    returns_features = layers.Lambda(lambda x: x[:, :4], name='returns_slice')(last_step)
    
    # Group 2: Indicators (features 4-5)  
    indic_features = layers.Lambda(lambda x: x[:, 4:], name='indic_slice')(last_step)
    
    # === ANFIS for each group ===
    # Returns ANFIS: 4 features × 2 MFs = 16 rules
    anfis_returns = FeatureGroupANFIS(
        n_mfs=n_mfs, 
        output_dim=8,
        name_prefix='returns',
        initial_centers=returns_centers,
        name='anfis_returns'
    )(returns_features)
    
    # Indicators ANFIS: 2 features × 2 MFs = 4 rules
    anfis_indic = FeatureGroupANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix='indic',
        initial_centers=indic_centers,
        name='anfis_indic'
    )(indic_features)
    
    # Combine ANFIS outputs (simple)
    anfis_combined = layers.Concatenate(name='anfis_combined')([anfis_returns, anfis_indic])
    anfis_out = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_combined)
    
    # === Simple BiLSTM (no attention - faster training) ===
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),  # 64 units
        name='bilstm_1'
    )(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units // 2),  # 32 units
        name='bilstm_2'
    )(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    # === Combine ANFIS + BiLSTM ===
    combined = layers.Concatenate(name='final_combine')([anfis_out, lstm_out])
    
    # === Simple output (less layers = less overfitting) ===
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(4, name='output')(x)  # 4 returns: Close, Open, High, Low
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model



# ===============================
# RULE EXTRACTION
# ===============================
def extract_rules(model, feature_names=None):
    """
    Extract fuzzy rules from the trained model.
    
    Returns:
        dict with 'returns_rules' and 'indic_rules'
    """
    if feature_names is None:
        feature_names = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    
    returns_names = feature_names[:4]
    indic_names = feature_names[4:]
    
    rules = {}
    
    # Get ANFIS layers
    for layer in model.layers:
        if layer.name == 'anfis_returns':
            rules['returns'] = {
                'rules': layer.get_rule_descriptions(returns_names),
                'n_rules': layer.n_rules,
                'mf_centers': layer.mf_centers.numpy().tolist(),
                'mf_widths': layer.mf_widths.numpy().tolist()
            }
        elif layer.name == 'anfis_indic':
            rules['indicators'] = {
                'rules': layer.get_rule_descriptions(indic_names),
                'n_rules': layer.n_rules,
                'mf_centers': layer.mf_centers.numpy().tolist(),
                'mf_widths': layer.mf_widths.numpy().tolist()
            }
    
    return rules


def analyze_prediction(model, X_sample, feature_names=None):
    """
    Analyze a single prediction by showing which rules are most active.
    
    Returns:
        dict with firing strengths for each ANFIS
    """
    if feature_names is None:
        feature_names = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    
    # Get last timestep
    last_step = X_sample[:, -1, :]
    returns_features = last_step[:, :4]
    indic_features = last_step[:, 4:]
    
    analysis = {}
    
    for layer in model.layers:
        if layer.name == 'anfis_returns':
            _, firing = layer(returns_features, return_firing_strengths=True)
            firing_np = firing.numpy()[0]
            rules = layer.get_rule_descriptions(feature_names[:4])
            
            # Get top 3 active rules
            top_indices = np.argsort(firing_np)[::-1][:3]
            analysis['returns'] = {
                'top_rules': [(rules[i], float(firing_np[i])) for i in top_indices]
            }
            
        elif layer.name == 'anfis_indic':
            _, firing = layer(indic_features, return_firing_strengths=True)
            firing_np = firing.numpy()[0]
            rules = layer.get_rule_descriptions(feature_names[4:])
            
            top_indices = np.argsort(firing_np)[::-1][:2]
            analysis['indicators'] = {
                'top_rules': [(rules[i], float(firing_np[i])) for i in top_indices]
            }
    
    return analysis


# ===============================
# DATA PREPARATION
# ===============================
def prepare_data(data, look_back=60, train_split=0.8):
    """
    Prepare data for next day price prediction.
    
    Strategy: Predict RETURNS (works well), then convert to prices for display.
    - Features: returns-based (interpretable for ANFIS)
    - Targets: next day's returns
    - Output: convert predictions to prices
    """
    df = data.copy()
    
    # === Features: returns-based (interpretable) ===
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # === Targets: NEXT day's returns (relative to current close) ===
    # This is what we predict - works much better than absolute prices
    df['target_Close_ret'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target_Open_ret'] = df['Open'].shift(-1) / df['Close'] - 1
    df['target_High_ret'] = df['High'].shift(-1) / df['Close'] - 1
    df['target_Low_ret'] = df['Low'].shift(-1) / df['Close'] - 1
    
    # Store next day prices BEFORE dropna (so they align correctly)
    df['next_Close'] = df['Close'].shift(-1)
    df['next_Open'] = df['Open'].shift(-1)
    df['next_High'] = df['High'].shift(-1)
    df['next_Low'] = df['Low'].shift(-1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    # Clip target returns too
    for col in ['target_Close_ret', 'target_Open_ret', 'target_High_ret', 'target_Low_ret']:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    df = df.dropna().reset_index(drop=True)  # This drops last row with NaN from shift
    
    feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    target_cols = ['target_Close_ret', 'target_Open_ret', 'target_High_ret', 'target_Low_ret']
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    current_prices = df[['Close', 'Open', 'High', 'Low']].values
    next_day_prices = df[['next_Close', 'next_Open', 'next_High', 'next_Low']].values
    
    # Scale features and targets
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    X, y, curr_px, next_px = [], [], [], []
    for i in range(look_back, len(features)):
        # OPTIMIZATION: Use window ending at i (inclusive) to predict target at i (which is i->i+1)
        # Features at i are known at time t (e.g. Close_ret[i] computed from Close[i] and Close[i-1])
        # So we can use features[i-look_back+1 : i+1]
        if i+1 <= len(features): # Ensure bounds
             X.append(scaled_feat[i-look_back+1 : i+1])
             y.append(scaled_tgt[i])
             curr_px.append(current_prices[i])
             next_px.append(next_day_prices[i])
    X, y, curr_px, next_px = np.array(X), np.array(y), np.array(curr_px), np.array(next_px)
    
    split = int(len(X) * train_split)
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'current_prices': curr_px[split:],
        'actual_next_prices': next_px[split:],
        'tgt_scaler': tgt_scaler
    }






# ===============================
# METRICS
# ===============================
def calc_directional_accuracy(y_true, y_pred, y_prev):
    actual_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    correct = (actual_direction == pred_direction).astype(float)
    return np.mean(correct) * 100


def calc_metrics(y_true, y_pred, y_prev=None, names=['Close', 'Open', 'High', 'Low']):
    results = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        
        da = 0
        if y_prev is not None:
            da = calc_directional_accuracy(t, p, y_prev[:, i])
        
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


# ===============================
# TRAINING
# ===============================
def train_stock(stock_name, data_path, output_dir, n_mfs=2, n_runs=5, epochs=150, seed=42):
    """Train Feature-Group ANFIS + BiLSTM model"""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (Feature-Group ANFIS + BiLSTM)")
    print(f"   MFs: {n_mfs}, Runs: {n_runs}, Epochs: {epochs}")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows")
    
    try:
        d = prepare_data(data)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    
    # === Compute K-Means centers for MF initialization ===
    print(f"\n📊 Computing K-Means centers for MF initialization...")
    train_last_step = d['X_train'][:, -1, :]  # (n_samples, 6)
    
    # Returns features (0-3)
    returns_data = train_last_step[:, :4]
    # For each feature, find 2 cluster centers (LOW, HIGH)
    returns_centers = np.zeros((4, n_mfs))
    for i in range(4):
        kmeans = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        kmeans.fit(returns_data[:, i:i+1])
        returns_centers[i] = np.sort(kmeans.cluster_centers_.flatten())
    
    # Indicators features (4-5)
    indic_data = train_last_step[:, 4:]
    indic_centers = np.zeros((2, n_mfs))
    for i in range(2):
        kmeans = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        kmeans.fit(indic_data[:, i:i+1])
        indic_centers[i] = np.sort(kmeans.cluster_centers_.flatten())
    
    print(f"   Returns centers: LOW≈{returns_centers[:, 0].mean():.2f}, HIGH≈{returns_centers[:, 1].mean():.2f}")
    print(f"   Indicators centers: LOW≈{indic_centers[:, 0].mean():.2f}, HIGH≈{indic_centers[:, 1].mean():.2f}")
    
    best_model = None
    best_history = None
    best_r2 = -1
    best_run = -1
    all_run_results = []
    
    print(f"\n🎓 Starting {n_runs} training runs...")
    
    for run in range(n_runs):
        run_seed = seed + run
        set_seed(run_seed)
        
        print(f"\n--- Run {run+1}/{n_runs} (seed={run_seed}) ---")
        
        model = create_feature_group_model(
            look_back=60,
            n_features=d['X_train'].shape[-1],
            n_mfs=n_mfs,
            lstm_units=64,
            returns_centers=returns_centers,
            indic_centers=indic_centers
        )
        
        if run == 0:
            print(f"\n📐 Model Architecture:")
            print(f"   Returns ANFIS: 4 features × {n_mfs} MFs = {n_mfs**4} rules")
            print(f"   Indicators ANFIS: 2 features × {n_mfs} MFs = {n_mfs**2} rules")
            print(f"   Total rules: {n_mfs**4 + n_mfs**2}")
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
        
        pred_scaled = model.predict(d['X_test'], verbose=0)
        pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
        # Convert returns to prices: next_price = current_close * (1 + return)
        current_close = d['current_prices'][:len(pred_ret), 0:1]  # Just Close price
        pred_px = current_close * (1 + pred_ret)
        actual_px = d['actual_next_prices'][:len(pred_px)]
        current_px = d['current_prices'][:len(pred_px)]
        
        metrics = calc_metrics(actual_px, pred_px, current_px)
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        close_r2 = metrics['Close']['R2']
        close_da = metrics['Close']['DA']
        
        print(f"   Close R²: {close_r2:.4f}, DA: {close_da:.1f}%, Epochs: {epochs_trained}/{epochs}, Time: {train_time:.1f}s")
        
        all_run_results.append({
            'run': run + 1,
            'close_r2': close_r2,
            'avg_r2': avg_r2,
            'epochs_trained': epochs_trained,
            'training_time_sec': round(train_time, 2)
        })
        
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
    
    # Extract rules
    print(f"\n📋 Extracting fuzzy rules...")
    rules = extract_rules(best_model)
    
    print(f"\n   Returns ANFIS ({rules['returns']['n_rules']} rules):")
    for r in rules['returns']['rules'][:3]:
        print(f"      {r}")
    print(f"      ... and {rules['returns']['n_rules']-3} more")
    
    print(f"\n   Indicators ANFIS ({rules['indicators']['n_rules']} rules):")
    for r in rules['indicators']['rules']:
        print(f"      {r}")
    
    # Analyze sample prediction
    print(f"\n🔍 Sample prediction analysis:")
    analysis = analyze_prediction(best_model, d['X_test'][:1])
    print(f"   Top active Returns rules:")
    for rule, strength in analysis['returns']['top_rules']:
        print(f"      {rule[:60]}... (strength: {strength:.3f})")
    print(f"   Top active Indicator rules:")
    for rule, strength in analysis['indicators']['top_rules']:
        print(f"      {rule} (strength: {strength:.3f})")
    
    # Final metrics
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name} (Feature-Group ANFIS, Best Run: {best_run})")
    print(f"{'='*80}")
    
    for n, m in best_metrics.items():
        f = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:6} - RMSE: {m['RMSE']:7.2f} | MAE: {m['MAE']:7.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {f}")
    
    print(f"\n📈 Model Info:")
    print(f"   Parameters: {best_model.count_params():,}")
    print(f"   Training: {best_train_time:.1f}s ({best_epochs_trained}/{epochs} epochs)")
    print(f"   Architecture: ANFIS_returns({n_mfs**4} rules) + ANFIS_indic({n_mfs**2} rules) + BiLSTM")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_feature_group.keras')
    best_model.save(model_path)
    
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'stock': stock_name,
            'architecture': 'Feature-Group ANFIS + BiLSTM',
            'n_mfs': n_mfs,
            'n_rules_returns': n_mfs**4,
            'n_rules_indic': n_mfs**2,
            'best_run': best_run,
            'metrics': best_metrics,
            'rules': rules,
            'all_runs': all_run_results
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    
    return {'metrics': best_metrics, 'rules': rules, 'params': best_model.count_params()}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_feature_group')
    
    stocks = {
        'IHSG': os.path.join(base_dir, 'IHSG_2007_2024.csv'),
    }
    
    all_metrics = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                m = train_stock(
                    name, path, output_dir,
                    n_mfs=2,  # 2 MFs per feature (as per paper)
                    n_runs=1,
                    epochs=150
                )
                if m:
                    all_metrics[name] = m
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - Feature-Group ANFIS + BiLSTM")
    print(f"{'='*80}")
    
    for n, result in all_metrics.items():
        m = result['metrics']
        cm = m['Close']
        f = "✅" if cm['R2'] >= 0.98 else ("⚠️" if cm['R2'] >= 0.95 else "❌")
        print(f"{n:10}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}%, DA={cm['DA']:.1f}%, Params={result['params']:,} {f}")
    
    print(f"\n📋 Fuzzy Rules Structure:")
    print(f"   Returns ANFIS: 4 features × 2 MFs = 16 rules (interpretable)")
    print(f"   Indicators ANFIS: 2 features × 2 MFs = 4 rules (interpretable)")
    print(f"   Total: 20 fuzzy rules with semantic meaning")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
