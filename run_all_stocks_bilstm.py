#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để train BiLSTM model với Fuzzy Logic cho nhiều stocks
Author: Auto-generated
Date: 2026-01-17
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
import warnings
warnings.filterwarnings('ignore')

# Thiết lập để chạy trên Mac M2 với Metal
try:
    # Kiểm tra GPU Metal
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Tìm thấy {len(gpus)} GPU(s): {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  Không tìm thấy GPU, sử dụng CPU")
except Exception as e:
    print(f"⚠️  Lỗi khi thiết lập GPU: {e}")

def prepare_fuzzy_features(data):
    """Tạo đặc trưng fuzzy từ dữ liệu giá"""
    # Tính biến x = (Close - Open) / (High - Low), xử lý chia cho 0
    data['x'] = data.apply(
        lambda row: (row['Close'] - row['Open']) / (row['High'] - row['Low'])
        if (row['High'] - row['Low']) != 0 else 0,
        axis=1
    )
    
    # Hàm chuẩn hóa x về khoảng [0,1]
    def normalize_x(x_series):
        min_val = x_series.min()
        max_val = x_series.max()
        range_val = max_val - min_val
        if range_val == 0:
            return x_series * 0
        return (x_series - min_val) / range_val
    
    # Thêm cột x_norm
    data['x_norm'] = normalize_x(data['x'])
    
    # GMM xây 5 hàm thuộc
    n_mfs = 5
    x = data['x_norm'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_mfs, random_state=0)
    gmm.fit(x)
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    sorted_idx = np.argsort(means)
    sorted_means = means[sorted_idx]
    sorted_stds = stds[sorted_idx]
    
    # Hàm Gaussian
    def gaussian(x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # Tính giá trị hàm thuộc cho toàn bộ data
    membership_matrix = np.zeros((len(data), n_mfs))
    for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
        membership_matrix[:, i] = gaussian(data['x_norm'].values, mean, std)
    
    # Sinh luật mờ Sugeno bậc nhất
    X_input = data[['Open', 'High', 'Low', 'Close']].values
    rule_outputs = []
    firing_strengths = []
    
    for i in range(n_mfs):
        w = membership_matrix[:, i]
        a = np.random.uniform(-1, 1, size=4)
        b = np.random.uniform(-0.5, 0.5)
        f = X_input @ a + b
        firing_strengths.append(w)
        rule_outputs.append(f)
    
    firing_strengths = np.stack(firing_strengths, axis=1)
    rule_outputs = np.stack(rule_outputs, axis=1)
    
    numerator = np.sum(firing_strengths * rule_outputs, axis=1)
    denominator = np.sum(firing_strengths, axis=1) + 1e-6
    data['y_outputfuzzy'] = numerator / denominator
    
    return data

def prepare_data(data, look_back=60, forecast_horizon=1, train_split=0.8):
    """Chuẩn bị dữ liệu cho BiLSTM"""
    price_data = data[['Close', 'Open', 'High', 'Low']].values
    fuzzy_data = data[['y_outputfuzzy']].values
    
    # Chuẩn hóa dữ liệu
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    fuzzy_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_prices = price_scaler.fit_transform(price_data)
    scaled_fuzzy = fuzzy_scaler.fit_transform(fuzzy_data)
    
    # Tạo dữ liệu với đầu vào là giá quá khứ và dự đoán fuzzy
    X, y = [], []
    for i in range(len(scaled_prices) - look_back - forecast_horizon + 1):
        price_features = scaled_prices[i:(i + look_back)]
        fuzzy_features = scaled_fuzzy[i:(i + look_back)]
        combined_features = np.column_stack((price_features, fuzzy_features))
        X.append(combined_features)
        y.append(scaled_prices[i + look_back + forecast_horizon - 1])
    
    X, y = np.array(X), np.array(y)
    
    # Tách tập huấn luyện và tập kiểm tra
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, price_scaler, fuzzy_scaler

def build_bilstm_model(input_shape, output_dim=4):
    """Xây dựng mô hình BiLSTM"""
    model = Sequential()
    
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    
    model.add(Dense(output_dim))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

def evaluate_model(model, X_test, y_test, price_scaler, stock_name):
    """Đánh giá mô hình"""
    y_pred = model.predict(X_test, verbose=0)
    
    y_test_actual = price_scaler.inverse_transform(y_test)
    y_pred_actual = price_scaler.inverse_transform(y_pred)
    
    metrics = {}
    price_types = ['Close', 'Open', 'High', 'Low']
    
    print(f"\n{'='*60}")
    print(f"📊 METRICS CHO {stock_name}")
    print(f"{'='*60}")
    
    for i, price_type in enumerate(price_types):
        mse = mean_squared_error(y_test_actual[:, i], y_pred_actual[:, i])
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test_actual[:, i], y_pred_actual[:, i])
        mape = np.mean(np.abs((y_test_actual[:, i] - y_pred_actual[:, i]) / y_test_actual[:, i])) * 100
        rmspe = np.sqrt(np.mean(np.square((y_test_actual[:, i] - y_pred_actual[:, i]) / y_test_actual[:, i]))) * 100
        r2 = r2_score(y_test_actual[:, i], y_pred_actual[:, i])
        
        metrics[price_type] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'RMSPE': rmspe,
            'R2': r2,
        }
        
        print(f"{price_type:6s} - RMSE: {rmse:7.4f} | MAE: {mae:7.4f} | MAPE: {mape:6.2f}% | R2: {r2:6.4f}")
    
    return metrics, y_test_actual, y_pred_actual

def train_and_evaluate(stock_name, data_path, output_dir, 
                       look_back=60, forecast_horizon=1, 
                       train_split=0.8, epochs=100, batch_size=32):
    """Train và đánh giá mô hình cho một stock"""
    
    print(f"\n{'#'*60}")
    print(f"🚀 BẮT ĐẦU TRAIN {stock_name}")
    print(f"{'#'*60}")
    
    # Tạo thư mục kết quả
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc dữ liệu
    print(f"📂 Đang đọc dữ liệu từ: {data_path}")
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    print(f"✅ Đã đọc {len(data)} dòng dữ liệu")
    
    # Tạo đặc trưng fuzzy
    print("🔮 Tạo đặc trưng fuzzy...")
    data = prepare_fuzzy_features(data)
    
    # Chuẩn bị dữ liệu
    print("🔧 Chuẩn bị dữ liệu...")
    X_train, X_test, y_train, y_test, price_scaler, fuzzy_scaler = prepare_data(
        data, look_back, forecast_horizon, train_split
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Xây dựng mô hình
    print("🏗️  Xây dựng mô hình BiLSTM...")
    model = build_bilstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=4)
    
    # Huấn luyện mô hình
    print(f"🎓 Bắt đầu huấn luyện (max {epochs} epochs)...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    print(f"✅ Hoàn thành sau {len(history.history['loss'])} epochs")
    
    # Vẽ biểu đồ training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stock_name} - Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{stock_name}_training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Đánh giá mô hình
    print("📈 Đánh giá mô hình...")
    metrics, y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, price_scaler, stock_name)
    
    # Lưu mô hình
    model_path = os.path.join(output_dir, f'{stock_name}_bilstm_model.h5')
    model.save(model_path)
    print(f"💾 Đã lưu mô hình tại: {model_path}")
    
    # Lưu metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_path = os.path.join(output_dir, f'{stock_name}_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print(f"💾 Đã lưu metrics tại: {metrics_path}")
    
    # Vẽ biểu đồ dự đoán
    test_dates = data['Date'].iloc[len(data) - len(y_test) - forecast_horizon + 1:].values
    price_types = ['Close', 'Open', 'High', 'Low']
    
    plt.figure(figsize=(15, 10))
    for i, price_type in enumerate(price_types):
        plt.subplot(2, 2, i+1)
        plt.plot(test_dates, y_test_actual[:, i], label=f'Actual {price_type}', alpha=0.7)
        plt.plot(test_dates, y_pred_actual[:, i], label=f'Predicted {price_type}', alpha=0.7)
        plt.title(f'{stock_name} - {price_type} Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ĐÃ HOÀN THÀNH {stock_name}\n")
    
    return metrics, model

def main():
    """Hàm chính"""
    # Danh sách các stocks cần train
    stocks = [
        {'name': 'AMZN', 'file': 'AMZN.csv'},
        {'name': 'JPM', 'file': 'JPM.csv'},
        {'name': 'TSLA', 'file': 'TSLA.csv'}
    ]
    
    # Thư mục chứa data
    data_dir = '/Users/bao/Documents/tsa_paper_1'
    
    # Tham số training
    params = {
        'look_back': 60,
        'forecast_horizon': 1,
        'train_split': 0.8,
        'epochs': 100,
        'batch_size': 32
    }
    
    # Dictionary lưu kết quả
    all_metrics = {}
    
    # Train từng stock
    for stock in stocks:
        stock_name = stock['name']
        data_path = os.path.join(data_dir, stock['file'])
        output_dir = os.path.join(data_dir, f'results_{stock_name}')
        
        try:
            metrics, model = train_and_evaluate(
                stock_name, data_path, output_dir, **params
            )
            all_metrics[stock_name] = metrics
        except Exception as e:
            print(f"❌ LỖI khi train {stock_name}: {str(e)}")
            continue
    
    # Tổng hợp kết quả
    print(f"\n{'='*60}")
    print("📊 TỔNG HỢP KẾT QUẢ")
    print(f"{'='*60}\n")
    
    summary = []
    for stock_name, metrics in all_metrics.items():
        close_metrics = metrics['Close']
        summary.append({
            'Stock': stock_name,
            'RMSE': close_metrics['RMSE'],
            'MAE': close_metrics['MAE'],
            'MAPE': close_metrics['MAPE'],
            'R2': close_metrics['R2']
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Lưu tổng hợp
    summary_path = os.path.join(data_dir, 'all_stocks_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Đã lưu tổng hợp tại: {summary_path}")
    
    print(f"\n{'='*60}")
    print("✨ HOÀN THÀNH TẤT CẢ!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
