#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAPTIVE Feature-Group ANFIS + BiLSTM
=====================================

Key Features:
1. Adaptive Rule Learning - dynamically prunes low-activation rules
2. Rule activation tracking during training
3. Data-driven number of effective rules

Based on: 
- Boyacioglu & Avci (2010) - ANFIS structure
- Alkharashi et al. (2025) - Adaptive rule reduction
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
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
# ADAPTIVE ANFIS LAYER
# ===============================
class AdaptiveANFIS(layers.Layer):
    """
    ANFIS with Adaptive Rule Learning.
    
    Features:
    1. Tracks rule activation (firing strength) during training
    2. Dynamically computes which rules are "active" vs "dormant"
    3. Can prune/mask low-activation rules
    4. Reports effective number of rules (data-driven)
    
    Based on: Alkharashi et al. (2025) - Self-Learning Type-2 Fuzzy
    """
    
    def __init__(self, n_mfs=2, output_dim=4, name_prefix='anfis', 
                 pruning_threshold=0.01, **kwargs):
        super(AdaptiveANFIS, self).__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.pruning_threshold = pruning_threshold  # Rules with avg firing < this are inactive
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.n_features = n_features
        self.n_rules = self.n_mfs ** n_features
        
        # MF parameters
        self.mf_centers = self.add_weight(
            name=f'{self.name_prefix}_mf_centers',
            shape=(n_features, self.n_mfs),
            initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
            trainable=True
        )
        
        self.mf_widths = self.add_weight(
            name=f'{self.name_prefix}_mf_widths',
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
        
        # Rule mask (for pruning) - non-trainable
        self.rule_mask = self.add_weight(
            name=f'{self.name_prefix}_rule_mask',
            shape=(self.n_rules,),
            initializer='ones',
            trainable=False
        )
        
        # Accumulated firing strengths (for tracking)
        self.accumulated_firing = self.add_weight(
            name=f'{self.name_prefix}_accumulated_firing',
            shape=(self.n_rules,),
            initializer='zeros',
            trainable=False
        )
        
        self.firing_count = self.add_weight(
            name=f'{self.name_prefix}_firing_count',
            shape=(),
            initializer='zeros',
            trainable=False
        )
        
        self.rule_mf_indices = self._compute_rule_indices(n_features, self.n_mfs)
        
        super(AdaptiveANFIS, self).build(input_shape)
    
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
    
    def call(self, inputs, training=False, return_firing_strengths=False):
        batch_size = tf.shape(inputs)[0]
        
        # Layer 1: Fuzzification
        x_exp = tf.expand_dims(inputs, 2)
        c = tf.expand_dims(self.mf_centers, 0)
        s = tf.abs(self.mf_widths) + 0.1
        s = tf.expand_dims(s, 0)
        
        memberships = tf.exp(-tf.square(x_exp - c) / (2 * tf.square(s)))
        
        # Layer 2: Rule Firing Strength
        firing_strengths = tf.ones((batch_size, self.n_rules))
        
        for feat_idx in range(self.n_features):
            mf_indices = self.rule_mf_indices[:, feat_idx]
            feat_memberships = memberships[:, feat_idx, :]
            rule_memberships = tf.gather(feat_memberships, mf_indices, axis=1)
            firing_strengths = firing_strengths * rule_memberships
        
        # Apply rule mask (prune inactive rules)
        firing_strengths = firing_strengths * self.rule_mask
        
        # Track firing during training
        if training:
            batch_mean_firing = tf.reduce_mean(firing_strengths, axis=0)
            self.accumulated_firing.assign_add(batch_mean_firing)
            self.firing_count.assign_add(1.0)
        
        # Layer 3: Normalize
        firing_sum = tf.reduce_sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        firing_norm = firing_strengths / firing_sum
        
        # Layer 4: TSK Consequents
        x_exp2 = tf.expand_dims(inputs, 1)
        x_exp3 = tf.expand_dims(x_exp2, 3)
        p_exp = tf.expand_dims(self.consequent_p, 0)
        
        linear = tf.reduce_sum(x_exp3 * p_exp, axis=2)
        rule_outputs = linear + self.consequent_r
        
        # Layer 5: Defuzzification
        firing_exp = tf.expand_dims(firing_norm, 2)
        output = tf.reduce_sum(firing_exp * rule_outputs, axis=1)
        
        if return_firing_strengths:
            return output, firing_norm
        return output
    
    def get_average_firing(self):
        """Get average firing strength per rule."""
        count = self.firing_count.numpy()
        if count == 0:
            return np.zeros(self.n_rules)
        return self.accumulated_firing.numpy() / count
    
    def get_active_rules(self):
        """Get indices of active rules (above threshold)."""
        avg_firing = self.get_average_firing()
        return np.where(avg_firing >= self.pruning_threshold)[0]
    
    def get_inactive_rules(self):
        """Get indices of inactive/dormant rules."""
        avg_firing = self.get_average_firing()
        return np.where(avg_firing < self.pruning_threshold)[0]
    
    def prune_rules(self, threshold=None):
        """Prune rules with firing strength below threshold."""
        if threshold is None:
            threshold = self.pruning_threshold
        
        avg_firing = self.get_average_firing()
        new_mask = (avg_firing >= threshold).astype(np.float32)
        self.rule_mask.assign(new_mask)
        
        n_active = int(np.sum(new_mask))
        n_pruned = self.n_rules - n_active
        
        return n_active, n_pruned
    
    def reset_tracking(self):
        """Reset firing strength tracking."""
        self.accumulated_firing.assign(tf.zeros(self.n_rules))
        self.firing_count.assign(0.0)
    
    def get_rule_statistics(self):
        """Get statistics about rule usage."""
        avg_firing = self.get_average_firing()
        mask = self.rule_mask.numpy()
        
        return {
            'total_rules': self.n_rules,
            'active_rules': int(np.sum(mask)),
            'dormant_rules': int(self.n_rules - np.sum(mask)),
            'avg_firing': avg_firing.tolist(),
            'max_firing': float(np.max(avg_firing)) if len(avg_firing) > 0 else 0,
            'min_firing': float(np.min(avg_firing[avg_firing > 0])) if np.any(avg_firing > 0) else 0,
            'rule_mask': mask.tolist()
        }
    
    def get_rule_descriptions(self, feature_names, include_firing=True):
        """Get rules with their activation status."""
        mf_labels = ['LOW', 'HIGH'] if self.n_mfs == 2 else ['LOW', 'MEDIUM', 'HIGH']
        avg_firing = self.get_average_firing()
        mask = self.rule_mask.numpy()
        
        rules = []
        for rule_idx in range(self.n_rules):
            conditions = []
            for feat_idx in range(self.n_features):
                mf_idx = self.rule_mf_indices[rule_idx, feat_idx].numpy()
                conditions.append(f"{feature_names[feat_idx]} is {mf_labels[mf_idx]}")
            
            status = "ACTIVE" if mask[rule_idx] > 0.5 else "DORMANT"
            rule_str = f"Rule {rule_idx+1}: IF " + " AND ".join(conditions)
            
            if include_firing:
                rule_str += f" [{status}, firing={avg_firing[rule_idx]:.4f}]"
            
            rules.append(rule_str)
        
        return rules
    
    def get_config(self):
        config = super(AdaptiveANFIS, self).get_config()
        config.update({
            'n_mfs': self.n_mfs,
            'output_dim': self.output_dim,
            'name_prefix': self.name_prefix,
            'pruning_threshold': self.pruning_threshold
        })
        return config


# ===============================
# RULE PRUNING CALLBACK
# ===============================
class AdaptiveRulePruningCallback(Callback):
    """
    Callback to prune low-activation rules during training.
    
    Strategy:
    - After warmup_epochs: analyze rule activations
    - Prune rules below threshold
    - Continue training with reduced rule set
    """
    
    def __init__(self, pruning_threshold=0.01, warmup_epochs=30, prune_every=50, verbose=True):
        super().__init__()
        self.pruning_threshold = pruning_threshold
        self.warmup_epochs = warmup_epochs
        self.prune_every = prune_every
        self.verbose = verbose
        self.pruning_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Only prune after warmup and at specified intervals
        if epoch < self.warmup_epochs:
            return
        
        if (epoch - self.warmup_epochs) % self.prune_every != 0:
            return
        
        total_pruned = 0
        
        for layer in self.model.layers:
            if isinstance(layer, AdaptiveANFIS):
                n_active, n_pruned = layer.prune_rules(self.pruning_threshold)
                total_pruned += n_pruned
                
                if self.verbose and n_pruned > 0:
                    print(f"\n   📉 Epoch {epoch+1}: Pruned {n_pruned} rules from {layer.name_prefix} "
                          f"(keeping {n_active}/{layer.n_rules})")
        
        self.pruning_history.append({
            'epoch': epoch + 1,
            'rules_pruned': total_pruned
        })


# ===============================
# MODEL CREATION
# ===============================
def create_adaptive_model(look_back, n_features, n_mfs=2, lstm_units=64, dropout=0.2, 
                          lr=0.001, pruning_threshold=0.01):
    """Create Adaptive Feature-Group ANFIS + BiLSTM model."""
    inputs = layers.Input(shape=(look_back, n_features), name='input')
    
    # Feature Groups
    last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(inputs)
    returns_features = layers.Lambda(lambda x: x[:, :4], name='returns_slice')(last_step)
    indic_features = layers.Lambda(lambda x: x[:, 4:], name='indic_slice')(last_step)
    
    # Adaptive ANFIS for each group
    anfis_returns = AdaptiveANFIS(
        n_mfs=n_mfs, 
        output_dim=8,
        name_prefix='returns',
        pruning_threshold=pruning_threshold,
        name='anfis_returns'
    )(returns_features)
    
    anfis_indic = AdaptiveANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix='indic',
        pruning_threshold=pruning_threshold,
        name='anfis_indic'
    )(indic_features)
    
    # Combine ANFIS outputs
    anfis_combined = layers.Concatenate(name='anfis_combined')([anfis_returns, anfis_indic])
    anfis_out = layers.Dense(16, activation='relu', name='anfis_dense')(anfis_combined)
    
    # BiLSTM
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm_1')(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units // 2), name='bilstm_2')(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    # Combine
    combined = layers.Concatenate(name='final_combine')([anfis_out, lstm_out])
    
    # Output
    x = layers.Dense(32, activation='relu', name='dense_1')(combined)
    x = layers.Dropout(dropout/2)(x)
    outputs = layers.Dense(4, name='output')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    
    return model


# ===============================
# HELPER FUNCTIONS
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
        'tgt_scaler': tgt_scaler
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
def train_stock(stock_name, data_path, output_dir, n_mfs=2, epochs=150, 
                pruning_threshold=0.01, seed=42):
    """Train Adaptive ANFIS model with rule pruning."""
    print(f"\n{'#'*70}")
    print(f"🚀 TRAINING {stock_name} (ADAPTIVE ANFIS + BiLSTM)")
    print(f"   MFs: {n_mfs}, Epochs: {epochs}, Pruning threshold: {pruning_threshold}")
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
    
    # Create model
    model = create_adaptive_model(
        look_back=60,
        n_features=d['X_train'].shape[-1],
        n_mfs=n_mfs,
        lstm_units=64,
        pruning_threshold=pruning_threshold
    )
    
    print(f"\n📐 Initial Architecture:")
    print(f"   Returns ANFIS: {n_mfs**4} rules (before pruning)")
    print(f"   Indicators ANFIS: {n_mfs**2} rules (before pruning)")
    print(f"   Parameters: {model.count_params():,}")
    
    # Callbacks with adaptive pruning
    pruning_callback = AdaptiveRulePruningCallback(
        pruning_threshold=pruning_threshold,
        warmup_epochs=30,  # Start pruning after epoch 30
        prune_every=50,    # Prune every 50 epochs
        verbose=True
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        LearningRateScheduler(lambda e: cosine_lr_schedule(e, 0.001, epochs)),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        pruning_callback
    ]
    
    # Train
    print(f"\n🎓 Training with adaptive rule pruning...")
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
    
    # Get rule statistics after training
    print(f"\n📊 Rule Statistics After Training:")
    rule_stats = {}
    feature_names = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret', 'range_pct', 'gap']
    
    for layer in model.layers:
        if isinstance(layer, AdaptiveANFIS):
            stats = layer.get_rule_statistics()
            layer_name = layer.name_prefix
            rule_stats[layer_name] = stats
            
            print(f"\n   {layer_name.upper()} ANFIS:")
            print(f"      Total rules: {stats['total_rules']}")
            print(f"      Active rules: {stats['active_rules']} ✅")
            print(f"      Dormant rules: {stats['dormant_rules']} 💤")
            print(f"      Max firing: {stats['max_firing']:.4f}")
            
            # Show active rules
            if layer_name == 'returns':
                rules = layer.get_rule_descriptions(feature_names[:4])
            else:
                rules = layer.get_rule_descriptions(feature_names[4:])
            
            active_rules = [r for r in rules if 'ACTIVE' in r]
            print(f"\n      Active rules:")
            for r in active_rules[:5]:
                print(f"         {r[:70]}...")
            if len(active_rules) > 5:
                print(f"         ... and {len(active_rules)-5} more")
    
    # Evaluate
    pred_scaled = model.predict(d['X_test'], verbose=0)
    pred_ret = d['tgt_scaler'].inverse_transform(pred_scaled)
    pred_px = d['base_prices'] * (1 + pred_ret)
    actual_px = d['actual_prices'][:len(pred_px)]
    y_prev_px = d['base_prices'][:len(pred_px)]
    
    metrics = calc_metrics(actual_px, pred_px, y_prev_px)
    
    # Final metrics
    print(f"\n{'='*80}")
    print(f"📊 FINAL METRICS FOR {stock_name} (Adaptive ANFIS)")
    print(f"{'='*80}")
    
    for n, m in metrics.items():
        f = "✅" if m['R2'] >= 0.98 else ("⚠️" if m['R2'] >= 0.95 else "❌")
        print(f"{n:6} - RMSE: {m['RMSE']:7.2f} | MAE: {m['MAE']:7.2f} | MAPE: {m['MAPE']:5.2f}% | R²: {m['R2']:.4f} | DA: {m['DA']:5.1f}% {f}")
    
    print(f"\n📈 Model Summary:")
    print(f"   Training: {train_time:.1f}s ({epochs_trained}/{epochs} epochs)")
    
    total_active = sum(s['active_rules'] for s in rule_stats.values())
    total_rules = sum(s['total_rules'] for s in rule_stats.values())
    print(f"   Effective rules: {total_active}/{total_rules} (data-driven!)")
    
    # Save
    model_path = os.path.join(output_dir, f'{stock_name}_adaptive.keras')
    model.save(model_path)
    
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'stock': stock_name,
            'architecture': 'Adaptive ANFIS + BiLSTM',
            'metrics': metrics,
            'rule_stats': rule_stats,
            'pruning_history': pruning_callback.pruning_history
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    
    return {'metrics': metrics, 'rule_stats': rule_stats}


def main():
    base_dir = '/Users/bao/Documents/tsa_paper_1'
    output_dir = os.path.join(base_dir, 'outputs_adaptive')
    
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
                    n_mfs=2,
                    epochs=150,
                    pruning_threshold=0.01,  # Rules with <1% activation are pruned
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
    print("📊 FINAL SUMMARY - Adaptive ANFIS + BiLSTM")
    print(f"{'='*80}")
    
    for name, result in all_results.items():
        m = result['metrics']
        cm = m['Close']
        stats = result['rule_stats']
        
        total_active = sum(s['active_rules'] for s in stats.values())
        total_rules = sum(s['total_rules'] for s in stats.values())
        
        f = "✅" if cm['R2'] >= 0.98 else ("⚠️" if cm['R2'] >= 0.95 else "❌")
        print(f"{name:10}: R²={cm['R2']:.4f}, DA={cm['DA']:.1f}%, Rules={total_active}/{total_rules} {f}")
    
    print("\n✨ COMPLETE!")


if __name__ == '__main__':
    main()
