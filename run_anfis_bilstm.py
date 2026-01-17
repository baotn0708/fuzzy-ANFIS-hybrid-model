#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM-ANFIS Hybrid for Stock Price Prediction
Combines:
- Returns-based prediction (stationary data)
- BiLSTM for temporal feature extraction
- Simplified ANFIS consequent layer (learnable rules)

"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
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


class ANFISLayer(layers.Layer):
    """
    Simplified ANFIS Layer based on paper's TSK approach.
    
    Uses fixed number of rules with learnable:
    - Rule centers (Gaussian MF centers)
    - Rule widths (Gaussian MF sigmas)  
    - Consequent weights (linear combination)
    
    Output = Σ(normalized_firing_strength_i × linear_consequent_i)
    """
    
    def __init__(self, n_rules=3, output_dim=4, **kwargs):
        super(ANFISLayer, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.output_dim = output_dim
    
    def build(self, input_shape):
        n_features = input_shape[-1]
        
        # Premise parameters: Gaussian MF centers and widths
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
        
        # Consequent parameters: linear weights for each rule
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
        # inputs: (batch, n_features)
        batch_size = tf.shape(inputs)[0]
        
        # Layer 1: Compute firing strength using Gaussian MF
        # Distance from each rule center
        x_exp = tf.expand_dims(inputs, 1)  # (batch, 1, features)
        c_exp = tf.expand_dims(self.centers, 0)  # (1, rules, features)
        
        # Squared distance
        dist_sq = tf.reduce_sum(tf.square(x_exp - c_exp), axis=-1)  # (batch, rules)
        
        # Gaussian firing strength
        sigma_sq = tf.square(tf.abs(self.widths) + 0.1)  # Ensure positive
        firing = tf.exp(-dist_sq / (2 * sigma_sq))  # (batch, rules)
        
        # Layer 2: Normalize firing strengths
        firing_sum = tf.reduce_sum(firing, axis=-1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum  # (batch, rules)
        
        # Layer 3: Compute rule outputs (first-order Sugeno)
        # For each rule: output_i = inputs @ W_i + b_i
        # inputs: (batch, features)
        # consequent_w: (rules, features, output_dim)
        
        x_exp2 = tf.expand_dims(inputs, 1)  # (batch, 1, features)
        x_exp3 = tf.expand_dims(x_exp2, -1)  # (batch, 1, features, 1)
        
        # Broadcast multiply and sum
        w_exp = tf.expand_dims(self.consequent_w, 0)  # (1, rules, features, output_dim)
        linear = tf.reduce_sum(x_exp3 * w_exp, axis=2)  # (batch, rules, output_dim)
        rule_outputs = linear + self.consequent_b  # (batch, rules, output_dim)
        
        # Layer 4: Weighted sum
        firing_exp = tf.expand_dims(firing_norm, -1)  # (batch, rules, 1)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)  # (batch, output_dim)
        
        return output
    
    def get_config(self):
        config = super(ANFISLayer, self).get_config()
        config.update({'n_rules': self.n_rules, 'output_dim': self.output_dim})
        return config


def create_bilstm_anfis_model(look_back, n_features, lstm_units=64, n_rules=3, dropout=0.2, lr=0.001):
    """
    BiLSTM-ANFIS Hybrid Model
    
    Architecture (based on paper):
    1. BiLSTM extracts temporal features
    2. ANFIS layer provides interpretable fuzzy rules for prediction
    """
    inputs = layers.Input(shape=(look_back, n_features))
    
    # BiLSTM for temporal feature extraction
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
    x = layers.Dropout(dropout)(x)
    
    # ANFIS layer for fuzzy inference
    outputs = ANFISLayer(n_rules=n_rules, output_dim=4, name='anfis')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    
    return model


def prepare_data(data, look_back=60, train_split=0.8):
    """Prepare returns-based data"""
    df = data.copy()
    
    # Calculate returns
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df = df.dropna()
    
    feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    target_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret']
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    prices = df[['Close', 'Open', 'High', 'Low']].values
    
    # Scale
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    # Create sequences
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


def calc_metrics(y_true, y_pred, names=['Close', 'Open', 'High', 'Low']):
    """Calculate metrics"""
    results = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mask = t != 0
        mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
        results[n] = {
            'RMSE': math.sqrt(mean_squared_error(t, p)),
            'MAE': mean_absolute_error(t, p),
            'MAPE': mape,
            'R2': r2_score(t, p)
        }
    return results


import yfinance as yf

# ... (API imports remain)

# [Retain existing classes: ANFISLayer, create_bilstm_anfis_model, prepare_data, calc_metrics, train_stock]
# NOTE: I am not replacing them, just ensuring the import is there. 
# Actually, I need to insert the import at the top and the download function. 
# I will use separate replace calls or a strategic single replacement if possible.
# But 'train_stock' calls 'pd.read_csv', I need to modify 'train_stock' to accept a DataFrame or handle download.

# Let's modify `train_stock` to handle data loading more flexibly and `main` to iterate tickers.

def get_stock_data(ticker_or_path, start_date='2020-01-01'):
    """Load data from local CSV or download from yfinance"""
    if os.path.exists(ticker_or_path):
        print(f"📂 Loading local file: {ticker_or_path}")
        df = pd.read_csv(ticker_or_path)
    else:
        print(f"📥 Downloading {ticker_or_path} from yfinance...")
        try:
            # increasing period to ensure enough data for 60-day lookback + split
            df = yf.Ticker(ticker_or_path).history(period="10y") 
            df = df.reset_index()
            # Ensure timezone-naive datetime for compatibility
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        except Exception as e:
            print(f"❌ Failed to download {ticker_or_path}: {e}")
            return None
            
    # Standardize column names if needed (yfinance uses Title Case)
    return df

def train_stock(stock_name, data_source, output_dir, n_rules=3):
    """Train BiLSTM-ANFIS model"""
    print(f"\n{'#'*60}")
    print(f"🚀 TRAIN {stock_name} (BiLSTM-ANFIS, {n_rules} rules)")
    print(f"{'#'*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = get_stock_data(data_source)
    if data is None or data.empty:
        print("❌ No data available")
        return None
        
    print(f"✅ Loaded {len(data)} rows")
    
    try:
        d = prepare_data(data)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return None
        
    print(f"   Train: {d['X_train'].shape}, Test: {d['X_test'].shape}")
    
    # Create model
    model = create_bilstm_anfis_model(
        look_back=60, n_features=d['X_train'].shape[-1],
        lstm_units=64, n_rules=n_rules
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]
    
    print(f"🎓 Training...")
    history = model.fit(
        d['X_train'], d['y_train'],
        validation_data=(d['X_test'], d['y_test']),
        epochs=100, batch_size=32,
        callbacks=callbacks, verbose=1
    )
    
    print(f"✅ Best val_loss: {min(history.history['val_loss']):.6f}")
    
    # Predict
    pred_scaled = model.predict(d['X_test'], verbose=0)
    pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
    # Be careful with dimensions matching
    base_prices = d['base_prices']
    pred_px = base_prices * (1 + pred_ret)
    
    actual_px = d['actual_prices'][:len(pred_px)]
    
    # Metrics
    metrics = calc_metrics(actual_px, pred_px)
    
    print(f"\n{'='*60}")
    print(f"📊 METRICS FOR {stock_name}")
    print(f"{'='*60}")
    
    for n, m in metrics.items():
        f = "✅" if m['R2'] > 0.9 and m['MAPE'] < 5 else "⚠️"
        print(f"{n:6} - RMSE: {m['RMSE']:8.2f} | MAE: {m['MAE']:8.2f} | MAPE: {m['MAPE']:5.2f}% | R2: {m['R2']:.4f} {f}")
    
    cm = metrics['Close']
    print(f"\n{'✅' if cm['R2'] >= 0.9 else '⚠️'} R² = {cm['R2']:.4f} (target: 0.9)")
    print(f"{'✅' if cm['MAPE'] <= 3 else '⚠️'} MAPE = {cm['MAPE']:.2f}% (target: 3%)")
    
    # Save model
    model.save(os.path.join(output_dir, f"{stock_name}_anfis.keras"))
    
    return metrics


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_yfinance')
    
    # List of stocks to test (Local + YFinance)
    stocks = {
        # Local files (keep for regression testing)
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        # New YFinance stocks
        'GOOGL': 'GOOGL',
        'MSFT': 'MSFT',
        'NVDA': 'NVDA',
        'AAPL': 'AAPL'
    }
    
    all_metrics = {}
    
    for name, source in stocks.items():
        try:
            m = train_stock(name, source, output_dir, n_rules=3)
            if m:
                all_metrics[name] = m
        except Exception as e:
            print(f"❌ ERROR {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 MỞ RỘNG KIỂM CHỨNG (YFINANCE)")
    print(f"{'='*60}")
    
    for n, m in all_metrics.items():
        cm = m['Close']
        f = "✅" if cm['R2'] > 0.9 and cm['MAPE'] <= 3 else "⚠️"
        print(f"{n:6}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}% {f}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
