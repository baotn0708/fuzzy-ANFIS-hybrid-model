#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data-Driven ANFIS + BiLSTM (True Adaptive)
==========================================

Based on papers:
1. peerj-cs-3004: Self-Learning Type-2 Fuzzy with Adaptive Rule Reduction
   - Rules generated from DATA, not fixed grid
   - KRLS for online parameter updates
   - Compatibility-based rule matching (threshold=0.9)
   
2. s41598-025-15022-8: ANFIS-ChHHO
   - Optimize ANFIS parameters with metaheuristic

Key Difference:
- Original ANFIS: n_mfs^n_features rules (fixed, exponential)
- Data-Driven: Rules created when input doesn't match existing (linear growth)

Algorithm:
1. Start with 0 rules
2. For each training sample:
   - Compute compatibility with existing rules
   - If max_compatibility < threshold: create new rule
   - Else: update most compatible rule using KRLS
3. Result: Minimal rule set that covers the data
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
# DATA-DRIVEN RULE GENERATION
# ===============================
class DataDrivenRuleGenerator:
    """
    Generate fuzzy rules from data using compatibility-based clustering.
    
    Based on Algorithm 1-3 from peerj-cs-3004:
    1. Rules are generated when input doesn't match existing rules
    2. Compatibility = Gaussian similarity to rule center
    3. If compatibility < threshold: create new rule
    
    This results in data-driven number of rules!
    """
    
    def __init__(self, compatibility_threshold=0.9, sigma=0.5):
        """
        Args:
            compatibility_threshold: Create new rule if max_compatibility < this
            sigma: Width of Gaussian compatibility function
        """
        self.threshold = compatibility_threshold
        self.sigma = sigma
        self.rule_centers = []  # List of (center, count)
        self.rule_outputs = []  # Average output for each rule
        
    def compute_compatibility(self, x, center):
        """
        Gaussian compatibility between input x and rule center.
        
        compatibility(x, c) = exp(-||x - c||^2 / (2*sigma^2))
        """
        dist_sq = np.sum((x - center) ** 2)
        return np.exp(-dist_sq / (2 * self.sigma ** 2))
    
    def fit(self, X, y, verbose=True):
        """
        Generate rules from training data.
        
        For each sample:
        - Find most compatible existing rule
        - If compatibility < threshold: create new rule
        - Else: update existing rule
        
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples, n_outputs)
        """
        n_samples = len(X)
        
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]
            
            if len(self.rule_centers) == 0:
                # First rule
                self.rule_centers.append({'center': xi.copy(), 'count': 1})
                self.rule_outputs.append(yi.copy())
                continue
            
            # Compute compatibility with all existing rules
            compatibilities = []
            for r in self.rule_centers:
                comp = self.compute_compatibility(xi, r['center'])
                compatibilities.append(comp)
            
            max_comp = max(compatibilities)
            best_rule_idx = np.argmax(compatibilities)
            
            if max_comp < self.threshold:
                # Create new rule
                self.rule_centers.append({'center': xi.copy(), 'count': 1})
                self.rule_outputs.append(yi.copy())
            else:
                # Update existing rule (incremental mean)
                r = self.rule_centers[best_rule_idx]
                n = r['count']
                r['center'] = (r['center'] * n + xi) / (n + 1)
                r['count'] = n + 1
                
                # Update output
                self.rule_outputs[best_rule_idx] = (
                    self.rule_outputs[best_rule_idx] * n + yi
                ) / (n + 1)
        
        if verbose:
            print(f"   Generated {len(self.rule_centers)} rules from {n_samples} samples")
        
        return self
    
    def get_rules(self):
        """Return rule centers and outputs."""
        centers = np.array([r['center'] for r in self.rule_centers])
        outputs = np.array(self.rule_outputs)
        counts = np.array([r['count'] for r in self.rule_centers])
        return centers, outputs, counts
    
    def get_rule_descriptions(self, feature_names):
        """Generate human-readable rule descriptions."""
        rules = []
        for i, r in enumerate(self.rule_centers):
            center = r['center']
            conditions = []
            for j, fname in enumerate(feature_names):
                val = center[j]
                if val < -0.5:
                    linguistic = "VERY_LOW"
                elif val < 0:
                    linguistic = "LOW"
                elif val < 0.5:
                    linguistic = "HIGH"
                else:
                    linguistic = "VERY_HIGH"
                conditions.append(f"{fname}≈{val:.2f}({linguistic})")
            
            rule_str = f"Rule {i+1} (n={r['count']}): IF " + " AND ".join(conditions)
            rules.append(rule_str)
        
        return rules


# ===============================
# DATA-DRIVEN ANFIS LAYER
# ===============================
class DataDrivenANFIS(layers.Layer):
    """
    ANFIS layer with data-driven rules.
    
    Instead of fixed M^N rules, uses rules discovered from data.
    Rule centers are initialized from DataDrivenRuleGenerator.
    """
    
    def __init__(self, initial_centers, initial_outputs, output_dim=4, 
                 name_prefix='ddanfis', **kwargs):
        """
        Args:
            initial_centers: Rule centers from DataDrivenRuleGenerator (n_rules, n_features)
            initial_outputs: Rule outputs (n_rules, n_outputs)  
            output_dim: Output dimension
        """
        super(DataDrivenANFIS, self).__init__(**kwargs)
        self.initial_centers = initial_centers
        self.initial_outputs = initial_outputs
        self.n_rules = len(initial_centers)
        self.output_dim = output_dim
        self.name_prefix = name_prefix
    
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        
        # Rule centers (initialized from data)
        self.centers = self.add_weight(
            name=f'{self.name_prefix}_centers',
            shape=(self.n_rules, n_features),
            initializer=tf.constant_initializer(self.initial_centers),
            trainable=True
        )
        
        # Rule widths (learnable)
        self.widths = self.add_weight(
            name=f'{self.name_prefix}_widths',
            shape=(self.n_rules,),
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )
        
        # TSK consequent: output = rule_output + linear_term
        # Initialize from data-derived outputs
        if self.initial_outputs.shape[1] != self.output_dim:
            # Pad or truncate
            out = np.zeros((self.n_rules, self.output_dim))
            out[:, :min(self.initial_outputs.shape[1], self.output_dim)] = \
                self.initial_outputs[:, :min(self.initial_outputs.shape[1], self.output_dim)]
        else:
            out = self.initial_outputs
            
        self.consequent_b = self.add_weight(
            name=f'{self.name_prefix}_consequent_b',
            shape=(self.n_rules, self.output_dim),
            initializer=tf.constant_initializer(out),
            trainable=True
        )
        
        # Linear terms (first-order TSK)
        self.consequent_w = self.add_weight(
            name=f'{self.name_prefix}_consequent_w',
            shape=(self.n_rules, n_features, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(DataDrivenANFIS, self).build(input_shape)
    
    def call(self, inputs, return_firing=False):
        """Forward pass."""
        # inputs: (batch, n_features)
        
        # Compute firing strengths (Gaussian)
        x_exp = tf.expand_dims(inputs, 1)  # (batch, 1, features)
        c = tf.expand_dims(self.centers, 0)  # (1, rules, features)
        
        # Euclidean distance
        diff = x_exp - c  # (batch, rules, features)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)  # (batch, rules)
        
        # Gaussian firing
        s = tf.abs(self.widths) + 0.1
        firing = tf.exp(-dist_sq / (2 * tf.square(s)))  # (batch, rules)
        
        # Normalize
        firing_sum = tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum  # (batch, rules)
        
        # TSK consequents: f = b + w*x
        x_exp2 = tf.expand_dims(inputs, 1)  # (batch, 1, features)
        x_exp3 = tf.expand_dims(x_exp2, -1)  # (batch, 1, features, 1)
        w_exp = tf.expand_dims(self.consequent_w, 0)  # (1, rules, features, out)
        
        linear = tf.reduce_sum(x_exp3 * w_exp, axis=2)  # (batch, rules, out)
        rule_outputs = linear + self.consequent_b  # (batch, rules, out)
        
        # Defuzzification
        firing_exp = tf.expand_dims(firing_norm, -1)  # (batch, rules, 1)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)  # (batch, out)
        
        if return_firing:
            return output, firing_norm
        return output
    
    def get_config(self):
        config = super(DataDrivenANFIS, self).get_config()
        config.update({
            'n_rules': self.n_rules,
            'output_dim': self.output_dim,
            'name_prefix': self.name_prefix
        })
        return config


# ===============================
# MODEL CREATION
# ===============================
def create_data_driven_model(look_back, n_features, rule_centers, rule_outputs,
                             lstm_units=64, dropout=0.2, lr=0.001):
    """
    Create model with data-driven ANFIS.
    
    Args:
        rule_centers: (n_rules, n_features) from DataDrivenRuleGenerator
        rule_outputs: (n_rules, 4) from DataDrivenRuleGenerator
    """
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # Get last timestep for ANFIS
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    
    # Data-Driven ANFIS
    n_rules = len(rule_centers)
    anfis_out = DataDrivenANFIS(
        initial_centers=rule_centers,
        initial_outputs=rule_outputs,
        output_dim=8,
        name_prefix='ddanfis',
        name='data_driven_anfis'
    )(last_step)
    
    anfis_processed = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_out)
    
    # BiLSTM
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm_1')(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units // 2), name='bilstm_2')(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    # Combine
    combined = layers.Concatenate(name='combine')([anfis_processed, lstm_out])
    
    # Output
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
    
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
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
    
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    X, y, base_px = [], [], []
    for i in range(look_back, len(features)):
        X.append(scaled_feat[i-look_back:i])
        y.append(scaled_tgt[i])
        base_px.append(prices[i-1])
    
    X, y, base_px = np.array(X), np.array(y), np.array(base_px)
    
    split = int(len(X) * train_split)
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'base_prices': base_px[split:],
        'actual_prices': prices[split + look_back:],
        'tgt_scaler': tgt_scaler,
        'scaled_feat_train': scaled_feat[:split + look_back],
        'scaled_tgt_train': scaled_tgt[:split + look_back]
    }


def calc_metrics(y_true, y_pred, y_prev=None, names=['Close', 'Open', 'High', 'Low']):
    results = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        
        da = 0
        if y_prev is not None:
            actual_dir = np.sign(t - y_prev[:, i])
            pred_dir = np.sign(p - y_prev[:, i])
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


# ===============================
# TRAINING
# ===============================
def train_stock(stock_name, data_path, output_dir, compatibility_threshold=0.9, 
                epochs=150, seed=42):
    """Train with Data-Driven ANFIS."""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (DATA-DRIVEN ANFIS + BiLSTM)")
    print(f"   Compatibility threshold: {compatibility_threshold}")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows")
    
    try:
        d = prepare_data(data)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return None
    
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    
    # === STEP 1: Generate Data-Driven Rules ===
    print(f"\n📋 Generating rules from data (threshold={compatibility_threshold})...")
    
    # Use last timestep features for rule generation
    train_features_last = d['X_train'][:, -1, :]  # (n_samples, 6)
    train_targets = d['y_train']  # (n_samples, 4)
    
    rule_gen = DataDrivenRuleGenerator(
        compatibility_threshold=compatibility_threshold,
        sigma=0.5
    )
    rule_gen.fit(train_features_last, train_targets, verbose=True)
    
    centers, outputs, counts = rule_gen.get_rules()
    n_rules = len(centers)
    
    print(f"   📊 Data-Driven Rules: {n_rules} (vs 2^6=64 for fixed grid)")
    
    # Show sample rules
    feature_names = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    rule_descs = rule_gen.get_rule_descriptions(feature_names)
    print(f"\n   Sample rules:")
    for r in rule_descs[:5]:
        print(f"      {r[:80]}...")
    if len(rule_descs) > 5:
        print(f"      ... and {len(rule_descs) - 5} more")
    
    # === STEP 2: Create Model ===
    print(f"\n🏗️ Creating model with {n_rules} data-driven rules...")
    
    model = create_data_driven_model(
        look_back=60,
        n_features=d['X_train'].shape[-1],
        rule_centers=centers,
        rule_outputs=outputs,
        lstm_units=64
    )
    
    print(f"   Parameters: {model.count_params():,}")
    
    # === STEP 3: Train ===
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        LearningRateScheduler(lambda e: cosine_lr_schedule(e, 0.001, epochs)),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
    ]
    
    print(f"\n🎓 Training...")
    train_start = time.time()
    history = model.fit(
        d['X_train'], d['y_train'],
        validation_data=(d['X_test'], d['y_test']),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - train_start
    epochs_trained = len(history.history['loss'])
    
    # === Evaluate ===
    pred_scaled = model.predict(d['X_test'], verbose=0)
    pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
    pred_px = d['base_prices'] * (1 + pred_ret)
    actual_px = d['actual_prices'][:len(pred_px)]
    y_prev_px = d['base_prices'][:len(pred_px)]
    
    metrics = calc_metrics(actual_px, pred_px, y_prev_px)
    
    # === Results ===
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name} (Data-Driven ANFIS)")
    print(f"{'='*80}")
    
    for n, m in metrics.items():
        f = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:6} - RMSE: {m['RMSE']:7.2f} | MAE: {m['MAE']:7.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {f}")
    
    print(f"\n📈 Model Summary:")
    print(f"   Data-Driven Rules: {n_rules} (created from data!)")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Training: {train_time:.1f}s ({epochs_trained}/{epochs} epochs)")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_data_driven.keras')
    model.save(model_path)
    
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'stock': stock_name,
            'architecture': 'Data-Driven ANFIS + BiLSTM',
            'n_rules_generated': n_rules,
            'n_rules_fixed_equiv': 2**6,
            'reduction_ratio': f"{n_rules} vs {2**6} ({100*n_rules/64:.1f}%)",
            'metrics': metrics,
            'rules': rule_descs
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    
    return {'metrics': metrics, 'n_rules': n_rules}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_data_driven')
    
    stocks = {
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        'JPM': os.path.join(base_dir, 'JPM.csv'),
        'TSLA': os.path.join(base_dir, 'TSLA.csv'),
    }
    
    all_results = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                result = train_stock(
                    name, path, output_dir,
                    compatibility_threshold=0.9,  # As per paper
                    epochs=150,
                    seed=42
                )
                if result:
                    all_results[name] = result
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - Data-Driven ANFIS + BiLSTM")
    print(f"{'='*80}")
    
    for name, result in all_results.items():
        m = result['metrics']
        cm = m['Close']
        n_rules = result['n_rules']
        
        f = "✅" if cm['R2'] >= 0.98 else ("⚠️" if cm['R2'] >= 0.95 else "❌")
        print(f"{name:10}: R²={cm['R2']:.4f}, DA={cm['DA']:.1f}%, Rules={n_rules} (data-driven!) {f}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
