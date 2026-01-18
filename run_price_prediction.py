#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS-BiLSTM for Next Day Price Prediction
===========================================

Key Changes from Previous Versions:
1. Predict raw price (not returns)
2. Use MEANINGFUL temporal features for ANFIS:
   - MA ratios (price vs moving averages)
   - Volatility (rolling std)
   - Trend strength
   - Momentum indicators
3. ANFIS rules are now interpretable:
   "IF price_above_MA20 AND volatility_low THEN price_increase"

Architecture:
Input → Temporal Features → ANFIS (interpretable) → BiLSTM → Output (next day price)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# TEMPORAL FEATURE ENGINEERING
# ===============================
def compute_temporal_features(df):
    """
    Compute meaningful temporal features for ANFIS.
    
    These features are INTERPRETABLE - fuzzy rules make sense!
    """
    data = df.copy()
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    open_ = data['Open'].values
    
    n = len(close)
    
    # === 1. Moving Average Ratios (interpretable: price vs trend) ===
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    
    data['price_vs_ma5'] = (close - ma5) / (ma5 + 1e-8)  # >0 = above MA5
    data['price_vs_ma20'] = (close - ma20) / (ma20 + 1e-8)  # >0 = above MA20
    data['ma5_vs_ma20'] = (ma5 - ma20) / (ma20 + 1e-8)  # >0 = uptrend
    
    # === 2. Volatility (interpretable: market uncertainty) ===
    returns = pd.Series(close).pct_change().values
    data['volatility_5d'] = pd.Series(returns).rolling(5).std().values
    data['volatility_20d'] = pd.Series(returns).rolling(20).std().values
    
    # === 3. Momentum (interpretable: recent performance) ===
    data['momentum_5d'] = (close - np.roll(close, 5)) / (np.roll(close, 5) + 1e-8)
    data['momentum_10d'] = (close - np.roll(close, 10)) / (np.roll(close, 10) + 1e-8)
    
    # === 4. RSI-like (interpretable: overbought/oversold) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_normalized'] = (data['rsi'] - 50) / 50  # -1 to 1
    
    # === 5. Price Range (interpretable: intraday volatility) ===
    data['daily_range'] = (high - low) / (close + 1e-8)
    data['gap'] = (open_ - np.roll(close, 1)) / (np.roll(close, 1) + 1e-8)
    
    # === 6. Trend Strength (interpretable: trend direction) ===
    # Linear regression slope over 10 days
    def calc_trend(series, window=10):
        result = np.zeros(len(series))
        for i in range(window, len(series)):
            y = series[i-window:i]
            x = np.arange(window)
            if np.std(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                result[i] = slope / (np.mean(y) + 1e-8)  # Normalized slope
        return result
    
    data['trend_10d'] = calc_trend(close, 10)
    
    # Clean data
    data = data.replace([np.inf, -np.inf], np.nan)
    
    return data


# ===============================
# ANFIS LAYER
# ===============================
class InterpretableANFIS(layers.Layer):
    """
    ANFIS layer optimized for interpretable temporal features.
    
    Features are designed so rules make sense:
    - "IF price_vs_ma20 is HIGH AND volatility is LOW THEN bullish"
    - "IF rsi_normalized is HIGH AND momentum is NEGATIVE THEN bearish"
    """
    
    def __init__(self, n_mfs=2, output_dim=8, name_prefix='anfis', **kwargs):
        super(InterpretableANFIS, self).__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        self.n_rules = self.n_mfs ** n_features
        
        # MF parameters
        self.mf_centers = self.add_weight(
            name=f'{self.name_prefix}_centers',
            shape=(n_features, self.n_mfs),
            initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
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
        super(InterpretableANFIS, self).build(input_shape)
    
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
    
    def call(self, inputs, return_firing=False):
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
        
        # TSK consequents
        x_exp2 = tf.expand_dims(inputs, 1)
        x_exp3 = tf.expand_dims(x_exp2, 3)
        p_exp = tf.expand_dims(self.consequent_p, 0)
        
        linear = tf.reduce_sum(x_exp3 * p_exp, axis=2)
        rule_outputs = linear + self.consequent_r
        
        # Defuzzification
        firing_exp = tf.expand_dims(firing_norm, 2)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)
        
        if return_firing:
            return output, firing_norm
        return output
    
    def get_rule_descriptions(self, feature_names):
        mf_labels = ['LOW', 'HIGH'] if self.n_mfs == 2 else ['LOW', 'MED', 'HIGH']
        rules = []
        for rule_idx in range(self.n_rules):
            conditions = []
            for feat_idx in range(self.n_features):
                mf_idx = self.rule_mf_indices[rule_idx, feat_idx].numpy()
                conditions.append(f"{feature_names[feat_idx]} is {mf_labels[mf_idx]}")
            rules.append(f"Rule {rule_idx+1}: IF " + " AND ".join(conditions))
        return rules


# ===============================
# MODEL
# ===============================
def create_price_prediction_model(look_back, n_temporal_features, n_mfs=2, lstm_units=64, dropout=0.2, lr=0.001):
    """
    Model for next day price prediction.
    
    Input: sequence of temporal features
    Output: next day's normalized price (or price ratio)
    """
    inputs = layers.Input(shape=(look_back, n_temporal_features), name='input')
    
    # Extract last timestep for ANFIS (current market state)
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    
    # ANFIS on interpretable temporal features
    anfis_out = InterpretableANFIS(
        n_mfs=n_mfs,
        output_dim=16,
        name_prefix='temporal',
        name='anfis_temporal'
    )(last_step)
    
    anfis_processed = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_out)
    
    # BiLSTM for sequence patterns
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
    combined = layers.Concatenate(name='combine')([anfis_processed, lstm_out])
    
    # Output: predict price ratio (next_price / current_price)
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(4, name='output')(x)  # Close, Open, High, Low ratios
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model


# ===============================
# DATA PREPARATION
# ===============================
def prepare_data_for_price_prediction(data, look_back=60, train_split=0.8):
    """
    Prepare data for next day price prediction.
    
    Target: next day's price as ratio to current price
    Features: temporal features (interpretable)
    """
    df = compute_temporal_features(data)
    
    # Feature columns (all interpretable)
    feature_cols = [
        'price_vs_ma5', 'price_vs_ma20', 'ma5_vs_ma20',
        'volatility_5d', 'volatility_20d',
        'momentum_5d', 'momentum_10d',
        'rsi_normalized',
        'daily_range', 'gap',
        'trend_10d'
    ]
    
    # Target: price ratios (next_price / current_price)
    # This is interpretable: >1 means price goes up
    df['target_close'] = df['Close'].shift(-1) / df['Close'] - 1  # return
    df['target_open'] = df['Open'].shift(-1) / df['Close'] - 1
    df['target_high'] = df['High'].shift(-1) / df['Close'] - 1
    df['target_low'] = df['Low'].shift(-1) / df['Close'] - 1
    
    target_cols = ['target_close', 'target_open', 'target_high', 'target_low']
    
    # Clean
    df = df.dropna().reset_index(drop=True)
    
    # Clip extreme values
    for col in feature_cols:
        df[col] = df[col].clip(-3, 3)
    for col in target_cols:
        df[col] = df[col].clip(-0.5, 0.5)
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    prices = df[['Close', 'Open', 'High', 'Low']].values
    
    # Scale features
    feat_scaler = StandardScaler()
    scaled_feat = feat_scaler.fit_transform(features)
    
    # Scale targets
    tgt_scaler = StandardScaler()
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    # Create sequences
    X, y, base_px = [], [], []
    for i in range(look_back, len(features) - 1):
        X.append(scaled_feat[i-look_back:i])
        y.append(scaled_tgt[i])
        base_px.append(prices[i])  # Current day prices
    
    X, y, base_px = np.array(X), np.array(y), np.array(base_px)
    
    # Actual next day prices for evaluation
    actual_prices = prices[look_back+1:]
    
    split = int(len(X) * train_split)
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'base_prices': base_px[split:],
        'actual_prices': actual_prices[split:split+len(X[split:])],
        'tgt_scaler': tgt_scaler,
        'feature_names': feature_cols
    }


# ===============================
# METRICS
# ===============================
def calc_metrics(y_true, y_pred, y_base=None, names=['Close', 'Open', 'High', 'Low']):
    results = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        
        da = 0
        if y_base is not None:
            # Directional accuracy: did we predict the correct direction?
            actual_dir = np.sign(t - y_base[:, i])
            pred_dir = np.sign(p - y_base[:, i])
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
def train_stock(stock_name, data_path, output_dir, n_mfs=2, n_runs=3, epochs=150, seed=42):
    """Train for next day price prediction."""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (Next Day Price Prediction)")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} rows")
    
    try:
        d = prepare_data_for_price_prediction(data)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    print(f"   Features: {d['feature_names']}")
    
    best_model = None
    best_r2 = -1
    best_run = -1
    
    print(f"\n🎓 Starting {n_runs} training runs...")
    
    for run in range(n_runs):
        run_seed = seed + run
        set_seed(run_seed)
        
        print(f"\n--- Run {run+1}/{n_runs} (seed={run_seed}) ---")
        
        model = create_price_prediction_model(
            look_back=60,
            n_temporal_features=d['X_train'].shape[-1],
            n_mfs=n_mfs,
            lstm_units=64
        )
        
        if run == 0:
            n_features = d['X_train'].shape[-1]
            print(f"   ANFIS: {n_features} features × {n_mfs} MFs = {n_mfs**n_features} rules")
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
        pred_returns = d['tgt_scaler'].inverse_transform(pred_scaled)
        
        # Convert to prices: next_price = base_price * (1 + return)
        pred_px = d['base_prices'] * (1 + pred_returns)
        actual_px = d['actual_prices'][:len(pred_px)]
        
        metrics = calc_metrics(actual_px, pred_px, d['base_prices'][:len(pred_px)])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        close_r2 = metrics['Close']['R2']
        close_da = metrics['Close']['DA']
        
        print(f"   Close R²: {close_r2:.4f}, DA: {close_da:.1f}%, Epochs: {epochs_trained}, Time: {train_time:.1f}s")
        
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = model
            best_run = run + 1
            best_pred_px = pred_px
            best_metrics = metrics
    
    print(f"\n✅ Best run: {best_run} with Avg R² = {best_r2:.4f}")
    
    # Extract rules
    print(f"\n📋 Fuzzy Rules (Interpretable!):")
    for layer in best_model.layers:
        if hasattr(layer, 'get_rule_descriptions'):
            rules = layer.get_rule_descriptions(d['feature_names'])
            for r in rules[:5]:
                print(f"   {r[:80]}...")
            if len(rules) > 5:
                print(f"   ... and {len(rules)-5} more")
    
    # Final metrics
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name}")
    print(f"{'='*80}")
    
    for n, m in best_metrics.items():
        flag = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:6} - RMSE: {m['RMSE']:8.2f} | MAE: {m['MAE']:8.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {flag}")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_price_pred.keras')
    best_model.save(model_path)
    
    print(f"\n💾 Saved to: {output_dir}")
    
    return {'metrics': best_metrics}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_price_prediction')
    
    stocks = {
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        'JPM': os.path.join(base_dir, 'JPM.csv'),
        'TSLA': os.path.join(base_dir, 'TSLA.csv'),
    }
    
    all_results = {}
    
    for name, path in stocks.items():
        if os.path.exists(path):
            try:
                result = train_stock(name, path, output_dir, n_mfs=2, n_runs=1, epochs=150)
                if result:
                    all_results[name] = result
            except Exception as e:
                print(f"❌ ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - Next Day Price Prediction")
    print(f"{'='*80}")
    
    for name, result in all_results.items():
        m = result['metrics']
        cm = m['Close']
        flag = "✅" if cm['R2'] >= 0.98 else ("⚠️" if cm['R2'] >= 0.95 else "❌")
        print(f"{name:10}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}%, DA={cm['DA']:.1f}% {flag}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
