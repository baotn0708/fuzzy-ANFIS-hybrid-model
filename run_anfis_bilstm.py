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


def create_bilstm_anfis_model(look_back, n_features, lstm_units=128, n_rules=5, dropout=0.2, lr=0.001):
    """
    AGGRESSIVE BiLSTM-ANFIS Hybrid Model
    
    Architecture upgrades for R² >= 0.98:
    1. Deeper BiLSTM: 128 → 64 → 32 units
    2. Multi-Head Attention layer 
    3. More ANFIS rules (5 instead of 3)
    """
    inputs = layers.Input(shape=(look_back, n_features))
    
    # === DEEP BiLSTM Stack ===
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    
    # === Multi-Head Attention ===
    # Focus on most important time steps
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])  # Residual connection
    x = layers.LayerNormalization()(x)
    
    # Final LSTM to compress
    x = layers.Bidirectional(layers.LSTM(lstm_units // 4))(x)
    x = layers.Dropout(dropout)(x)
    
    # Dense layer before ANFIS
    x = layers.Dense(64, activation='relu')(x)
    
    # ANFIS layer for fuzzy inference
    outputs = ANFISLayer(n_rules=n_rules, output_dim=4, name='anfis')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    
    return model


def prepare_data(data, look_back=60, train_split=0.8):
    """
    Returns-Based Data Preparation for Meaningful ANFIS.
    
    Why Returns?
    - ANFIS fuzzy rules can classify meaningful patterns:
      "Large Positive Return" (-5% to +5% range with fuzzy membership)
      "Small Negative", "Neutral", etc.
    - Raw prices normalized to [0,1] lose this semantic meaning.
    
    Features:
    - OHLC Returns (% change)
    - RSI (already [0,100], rescaled to [0,1])
    - MACD (normalized)
    - BB%B (already ~[0,1])
    - range_pct, gap
    """
    df = data.copy()
    
    # === OHLC Returns (Core for ANFIS) ===
    for col in ['Close', 'Open', 'High', 'Low']:
        df[f'{col}_ret'] = df[col].pct_change()
    
    # === Technical Indicators ===
    # RSI (14-period) - scale to [0,1]
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = (100 - (100 / (1 + rs))) / 100  # Scale to [0,1]
    
    # MACD - will be normalized per-window
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # Bollinger Band %B - already ~[0,1]
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_pctB'] = (df['Close'] - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    
    # Range and Gap (from original successful model)
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    # Features: Returns + Indicators
    feature_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 
                    'RSI', 'MACD', 'BB_pctB', 'range_pct', 'gap']
    target_cols = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret']
    
    features = df[feature_cols].values
    targets = df[target_cols].values
    prices = df[['Close', 'Open', 'High', 'Low']].values
    
    # Scale features (StandardScaler for returns works well)
    feat_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    
    scaled_feat = feat_scaler.fit_transform(features)
    scaled_tgt = tgt_scaler.fit_transform(targets)
    
    # Create sequences
    X, y, base_px = [], [], []
    for i in range(look_back, len(features)):
        X.append(scaled_feat[i-look_back:i])
        y.append(scaled_tgt[i])
        base_px.append(prices[i-1])  # Previous day's prices for recovering actual
    
    X, y, base_px = np.array(X), np.array(y), np.array(base_px)
    
    split = int(len(X) * train_split)
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:],
        'base_prices': base_px[split:],
        'actual_prices': prices[look_back:][split:],
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
    
    # Create AGGRESSIVE model (Returns-based with meaningful ANFIS)
    model = create_bilstm_anfis_model(
        look_back=60, n_features=d['X_train'].shape[-1],
        lstm_units=128, n_rules=5
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    print(f"🎓 Training (Aggressive: 200 epochs)...")
    history = model.fit(
        d['X_train'], d['y_train'],
        validation_data=(d['X_test'], d['y_test']),
        epochs=200, batch_size=32,
        callbacks=callbacks, verbose=1
    )
    
    print(f"✅ Best val_loss: {min(history.history['val_loss']):.6f}")
    
    # Predict
    pred_scaled = model.predict(d['X_test'], verbose=0)
    
    # Denormalize: scaled returns → actual returns
    pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
    
    # Convert returns to prices: price = base_price * (1 + return)
    base_prices = d['base_prices']
    pred_px = base_prices * (1 + pred_ret)
    actual_px = d['actual_prices']
    
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
    
    stocks = {
        # Local Datasets
        'AMZN': os.path.join(base_dir, 'AMZN.csv'),
        'JPM': os.path.join(base_dir, 'JPM.csv'),
        'TSLA': os.path.join(base_dir, 'TSLA.csv'),
        'FINAL_USO (Gold)': os.path.join(base_dir, 'FINAL_USO.csv'),
        
        # YFinance Datasets
        'GOOGL': 'GOOGL',
        'MSFT': 'MSFT',
        'NVDA': 'NVDA',
        'AAPL': 'AAPL',
        'Gold (YF)': 'GC=F'
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
        f = "✅" if cm['R2'] > 0.98 and cm['MAPE'] <= 3 else "⚠️"
        print(f"{n:6}: R²={cm['R2']:.4f}, MAPE={cm['MAPE']:.2f}% {f}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
