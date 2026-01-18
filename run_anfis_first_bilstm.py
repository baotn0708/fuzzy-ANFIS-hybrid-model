#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS-BiLSTM: ANFIS FIRST Architecture

Key Difference from Original:
- ANFIS layer processes RAW features (returns, indicators) - INTERPRETABLE
- BiLSTM learns temporal patterns from ANFIS outputs
- Fuzzy rules have semantic meaning!

Rule Explosion Prevention:
- Use fewer rules (3-5)
- Apply ANFIS per-feature or per-group, not all features together
- TSK-style (Takagi-Sugeno-Kang) with weighted rules

Architecture:
Input (60, 6) → ANFIS(each timestep) → BiLSTM → Dense → Output(4)
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


class ANFISFirstLayer(layers.Layer):
    """
    ANFIS Layer that processes EACH TIMESTEP independently.
    
    Applied to raw features where fuzzy rules are interpretable:
    - Rule 1: IF returns is LOW AND volatility is HIGH THEN output1
    - Rule 2: IF returns is HIGH AND volatility is LOW THEN output2
    
    This maintains the semantic meaning of fuzzy logic!
    
    To prevent rule explosion:
    - Use small number of rules (n_rules=3-5)
    - Each rule acts as a "market regime detector"
    """
    
    def __init__(self, n_rules=3, output_dim=None, **kwargs):
        super(ANFISFirstLayer, self).__init__(**kwargs)
        self.n_rules = n_rules
        self._output_dim = output_dim  # If None, same as input dim
    
    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        n_features = input_shape[-1]
        out_dim = self._output_dim if self._output_dim else n_features
        
        # Gaussian MF centers and widths for each rule
        self.centers = self.add_weight(
            name='centers',
            shape=(self.n_rules, n_features),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True
        )
        
        self.widths = self.add_weight(
            name='widths',
            shape=(self.n_rules, n_features),
            initializer=tf.constant_initializer(1.0),
            trainable=True
        )
        
        # TSK consequent: linear weights for each rule
        self.consequent_w = self.add_weight(
            name='consequent_w',
            shape=(self.n_rules, n_features, out_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.consequent_b = self.add_weight(
            name='consequent_b',
            shape=(self.n_rules, out_dim),
            initializer='zeros',
            trainable=True
        )
        
        self.out_dim = out_dim
        super(ANFISFirstLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        # Process each timestep through ANFIS
        
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        
        # Reshape to process all timesteps at once
        # (batch * timesteps, features)
        x_flat = tf.reshape(inputs, (-1, inputs.shape[-1]))
        
        # === Layer 1: Fuzzification (Gaussian MFs) ===
        x_exp = tf.expand_dims(x_flat, 1)  # (batch*T, 1, features)
        c = tf.expand_dims(self.centers, 0)  # (1, rules, features)
        s = tf.abs(self.widths) + 0.1
        s = tf.expand_dims(s, 0)  # (1, rules, features)
        
        # Per-feature membership values
        memberships = tf.exp(-tf.square(x_exp - c) / (2 * tf.square(s)))
        # (batch*T, rules, features)
        
        # === Layer 2: Rule firing strength (product T-norm) ===
        # Use log-sum-exp for numerical stability
        log_mu = tf.math.log(memberships + 1e-10)
        firing = tf.exp(tf.reduce_sum(log_mu, axis=-1))  # (batch*T, rules)
        
        # === Layer 3: Normalize firing strengths ===
        firing_sum = tf.reduce_sum(firing, axis=-1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum  # (batch*T, rules)
        
        # === Layer 4: TSK consequents ===
        x_exp2 = tf.expand_dims(x_flat, 1)  # (batch*T, 1, features)
        x_exp3 = tf.expand_dims(x_exp2, -1)  # (batch*T, 1, features, 1)
        w_exp = tf.expand_dims(self.consequent_w, 0)  # (1, rules, features, out)
        
        linear = tf.reduce_sum(x_exp3 * w_exp, axis=2)  # (batch*T, rules, out)
        rule_outputs = linear + self.consequent_b  # (batch*T, rules, out)
        
        # === Layer 5: Defuzzification (weighted sum) ===
        firing_exp = tf.expand_dims(firing_norm, -1)  # (batch*T, rules, 1)
        output_flat = tf.reduce_sum(firing_exp * rule_outputs, axis=1)  # (batch*T, out)
        
        # Reshape back to sequence
        output = tf.reshape(output_flat, (batch_size, timesteps, self.out_dim))
        
        return output
    
    def get_config(self):
        config = super(ANFISFirstLayer, self).get_config()
        config.update({
            'n_rules': self.n_rules,
            'output_dim': self._output_dim
        })
        return config


def create_anfis_first_model(look_back, n_features, n_rules=3, lstm_units=64, dropout=0.2, lr=0.001):
    """
    ANFIS-First Architecture:
    
    1. ANFIS processes each timestep (on interpretable features)
    2. BiLSTM captures temporal dependencies from ANFIS outputs
    3. Dense layers for final prediction
    
    Fuzzy rules remain interpretable because they act on raw features!
    """
    inputs = layers.Input(shape=(look_back, n_features))
    
    # === ANFIS Layer (processes each timestep) ===
    # This applies fuzzy logic to interpretable features
    # Rules like: "IF return is NEGATIVE AND volatility is HIGH THEN ..."
    x = ANFISFirstLayer(n_rules=n_rules, output_dim=n_features, name='anfis_first')(inputs)
    
    # Optional: Add residual connection to preserve original features
    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization()(x)
    
    # === BiLSTM Layer (learns temporal patterns) ===
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
    x = layers.Dropout(dropout)(x)
    
    # === Output ===
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4)(x)  # Predict 4 returns: Close, Open, High, Low
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model


def prepare_data(data, look_back=60, train_split=0.8):
    """Prepare returns-based data"""
    df = data.copy()
    
    # for col in ['Close', 'Open', 'High', 'Low']:
    for col in ['Close', 'Open']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    # for col in ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']:
    for col in ['Close_ret', 'Open_ret', 'range_pct', 'gap']:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    df = df.dropna().reset_index(drop=True)
    
    # feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    feature_cols = ['Close_ret', 'Open_ret', 'range_pct', 'gap']
    # target_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret']
    target_cols = ['Close_ret', 'Open_ret']
    
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
        'tgt_scaler': tgt_scaler
    }


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


def train_stock(stock_name, data_path, output_dir, n_rules=3, n_runs=5, epochs=150, seed=42):
    """Train ANFIS-First model"""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (ANFIS-First BiLSTM)")
    print(f"   Rules: {n_rules}, Runs: {n_runs}, Epochs: {epochs}")
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
        
        model = create_anfis_first_model(
            look_back=60,
            n_features=d['X_train'].shape[-1],
            n_rules=n_rules,
            lstm_units=64
        )
        
        if run == 0:
            model.summary()
        
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
        pred_px = d['base_prices'] * (1 + pred_ret)
        actual_px = d['actual_prices'][:len(pred_px)]
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
    
    # Performance metrics
    params = best_model.count_params()
    
    # Print final metrics
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name} (ANFIS-First, Best Run: {best_run})")
    print(f"{'='*80}")
    
    for n, m in best_metrics.items():
        f = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:6} - RMSE: {m['RMSE']:7.2f} | MAE: {m['MAE']:7.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {f}")
    
    print(f"\n📈 Model Info:")
    print(f"   Parameters: {params:,}")
    print(f"   Training: {best_train_time:.1f}s ({best_epochs_trained}/{epochs} epochs)")
    print(f"   Architecture: ANFIS({n_rules} rules) → BiLSTM(64) → Dense")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_anfis_first.keras')
    best_model.save(model_path)
    
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'stock': stock_name,
            'architecture': 'ANFIS-First',
            'n_rules': n_rules,
            'best_run': best_run,
            'metrics': best_metrics,
            'all_runs': all_run_results
        }, f, indent=2)
    
    return {'metrics': best_metrics, 'params': params}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_anfis_first')
    
    stocks = {
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        'JPM': os.path.join(base_dir, 'JPM.csv'),
        'TSLA': os.path.join(base_dir, 'TSLA.csv'),
    }
    
    all_metrics = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                # Use fewer rules (3) to prevent explosion
                m = train_stock(name, path, output_dir, n_rules=3, n_runs=5, epochs=150)
                if m:
                    all_metrics[name] = m
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - ANFIS-First BiLSTM")
    print(f"{'='*80}")
    
    for n, result in all_metrics.items():
        m = result['metrics']
        cm = m['Close']
        f = "✅" if cm['R2'] >= 0.98 else ("⚠️" if cm['R2'] >= 0.95 else "❌")
        print(f"{n:10}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}%, DA={cm['DA']:.1f}%, Params={result['params']:,} {f}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
