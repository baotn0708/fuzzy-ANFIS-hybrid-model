# Sơ đồ mô hình `run_feature_group_anfis_clean.py`

Tài liệu này bám sát phiên bản mới nhất của mô hình trong file `run_feature_group_anfis_clean.py`, chủ yếu theo các hàm:

- `build_model(...)`
- `reconstruct_ohlc(...)`
- `build_decompose_model(...)`
- `analyze_sample(...)`

## Sơ đồ tổng quan

```mermaid
flowchart LR
    A["Đầu vào chuỗi<br/>sequence_input<br/>shape: (batch, look_back, n_seq_features)"]

    A --> B["Lấy bước cuối của 6 đặc trưng lõi<br/>last_core_features"]
    B --> C["returns_slice<br/>4 đặc trưng:<br/>close_ret, open_gap,<br/>high_buffer, low_buffer"]
    B --> D["indicator_slice<br/>2 đặc trưng:<br/>range_pct, volume_ret"]

    C --> E["ANFIS returns<br/>output_dim = 4<br/>số luật = n_mfs^4"]
    D --> F["ANFIS indicators<br/>output_dim = 4<br/>số luật = n_mfs^2"]

    E --> G["Ghép đặc trưng mờ<br/>fuzzy_concat"]
    F --> G
    G --> H["Dense(16, tanh)<br/>fuzzy_hidden"]
    H --> I["Dense(4)<br/>fuzzy_raw_params"]

    A --> J["BiLSTM 1<br/>return_sequences = True"]
    J --> K["Dropout"]
    K --> L["BiLSTM 2"]
    L --> M["Dropout"]
    M --> N["Dense(16, relu)<br/>sequence_hidden"]
    N --> O["Dense(4)<br/>sequence_residual_params"]

    I --> P["Cộng hai nhánh<br/>raw_target_params"]
    O --> P

    P --> Q1["close_ret_output<br/>RETURN_CLIP * tanh(z0)"]
    P --> Q2["open_gap_output<br/>RETURN_CLIP * tanh(z1)"]
    P --> Q3["high_buffer_output<br/>BUFFER_CLIP * sigmoid(z2)"]
    P --> Q4["low_buffer_output<br/>BUFFER_CLIP * sigmoid(z3)"]

    Q1 --> R["Ghép 4 tham số mục tiêu<br/>target_output"]
    Q2 --> R
    Q3 --> R
    Q4 --> R

    R --> S["Dựng lại OHLC hợp lệ"]
```

## Sơ đồ chi tiết toàn pipeline

```mermaid
flowchart TD
    subgraph P0["1. Tiền xử lý dữ liệu"]
        D0["CSV trong Dataset/"]
        D1["load_market_dataframe<br/>- kiểm tra OHLC<br/>- parse Date nếu có<br/>- sort theo thời gian<br/>- ép kiểu số"]
        D2["engineer_features<br/>- tạo 6 đặc trưng lõi<br/>- tạo 4 target tham số hóa<br/>- giữ next_Open/High/Low/Close để đánh giá"]
        D3["build_windows<br/>- tạo rolling windows<br/>- chia train/val/test theo thời gian<br/>- fit scaler chỉ trên train<br/>- transform val/test bằng thống kê train"]
        D0 --> D1 --> D2 --> D3
    end

    subgraph P1["2. Đầu vào mô hình"]
        X0["X_train / X_val / X_test<br/>shape: (batch, look_back, n_seq_features)"]
        X1["y_train / y_val / y_test<br/>shape: (batch, 4)"]
    end

    subgraph P2["3. Nhánh mờ"]
        F0["last_core_features<br/>x[:, -1, :6]"]
        F1["returns_slice<br/>x[:, :4]"]
        F2["indicator_slice<br/>x[:, 4:6]"]
        F3["anfis_returns<br/>4 đầu vào -> 4 đầu ra ẩn"]
        F4["anfis_indicators<br/>2 đầu vào -> 4 đầu ra ẩn"]
        F5["fuzzy_concat"]
        F6["Dense(16, tanh)"]
        F7["Dense(4)<br/>fuzzy_raw_params"]
        F0 --> F1 --> F3
        F0 --> F2 --> F4
        F3 --> F5
        F4 --> F5
        F5 --> F6 --> F7
    end

    subgraph P3["4. Nhánh chuỗi"]
        S0["sequence_input<br/>toàn bộ cửa sổ"]
        S1["BiLSTM 1"]
        S2["Dropout 1"]
        S3["BiLSTM 2"]
        S4["Dropout 2"]
        S5["Dense(16, relu)"]
        S6["Dense(4)<br/>sequence_residual_params"]
        S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    subgraph P4["5. Kết hợp đầu ra"]
        C0["raw_target_params<br/>= fuzzy_raw_params + sequence_residual_params"]
        C1["close_ret_output<br/>RETURN_CLIP * tanh"]
        C2["open_gap_output<br/>RETURN_CLIP * tanh"]
        C3["high_buffer_output<br/>BUFFER_CLIP * sigmoid"]
        C4["low_buffer_output<br/>BUFFER_CLIP * sigmoid"]
        C5["target_output<br/>(close_ret, open_gap, high_buffer, low_buffer)"]
        C0 --> C1
        C0 --> C2
        C0 --> C3
        C0 --> C4
        C1 --> C5
        C2 --> C5
        C3 --> C5
        C4 --> C5
    end

    subgraph P5["6. Dựng lại giá và đánh giá"]
        R0["reconstruct_ohlc(current_close, target_output)"]
        R1["pred_close = current_close * (1 + close_ret)"]
        R2["pred_open = current_close * (1 + open_gap)"]
        R3["body_high = max(pred_open, pred_close)"]
        R4["body_low = min(pred_open, pred_close)"]
        R5["pred_high = body_high * exp(high_buffer)"]
        R6["pred_low = body_low * exp(-low_buffer)"]
        R7["predicted OHLC = [Open, High, Low, Close]"]
        R8["evaluate_predictions<br/>- RMSE, MAE, MAPE, R2<br/>- độ chính xác hướng<br/>- tỉ lệ OHLC hợp lệ"]
        R0 --> R1
        R0 --> R2
        R1 --> R3
        R2 --> R3
        R1 --> R4
        R2 --> R4
        R3 --> R5
        R4 --> R6
        R5 --> R7
        R6 --> R7
        R7 --> R8
    end

    D3 --> X0
    D3 --> X1
    X0 --> F0
    X0 --> S0
    F7 --> C0
    S6 --> C0
    C5 --> R0
```

## Sơ đồ chi tiết bên trong một khối `OrderedFeatureGroupANFIS`

```mermaid
flowchart TD
    A["Đầu vào cục bộ<br/>inputs shape: (batch, n_features)"]
    B["Các tâm có thứ tự<br/>center_base + softplus(center_delta_raw)"]
    C["Độ rộng Gaussian dương<br/>softplus(width_raw)"]
    D["Mờ hóa Gaussian<br/>memberships[b, i, j]"]
    E["rule_mf_indices<br/>xác định luật nào dùng hàm thuộc nào"]
    F["Cường độ kích hoạt luật<br/>firing = tích các độ thuộc"]
    G["Chuẩn hóa<br/>firing_norm = firing / tổng firing"]
    H["Hệ quả Sugeno<br/>rule_output = bias + tổng(coeff_i * x_i)"]
    I["Cộng có trọng số theo luật<br/>output = tổng(firing_norm * rule_output)"]
    J["Vector ẩn đầu ra của khối ANFIS"]

    A --> D
    B --> D
    C --> D
    D --> F
    E --> F
    F --> G
    A --> H
    G --> I
    H --> I
    I --> J
```

## Sơ đồ phục vụ giải thích nội bộ

```mermaid
flowchart LR
    A["sample window"] --> B["decompose_model"]
    B --> C["fuzzy_raw_params<br/>đầu ra thô của nhánh mờ"]
    B --> D["sequence_residual_params<br/>phần hiệu chỉnh của nhánh chuỗi"]
    B --> E["target_output<br/>đầu ra cuối cùng sau khi bị chặn"]

    A --> F["Lấy last_core"]
    F --> G["anfis_returns(return_details=True)"]
    F --> H["anfis_indicators(return_details=True)"]
    G --> I["returns_firing + returns_rule_outputs"]
    H --> J["indicators_firing + indicators_rule_outputs"]
    I --> K["Xếp hạng luật theo contribution_score"]
    J --> K
```

## Tóm tắt tensor quan trọng

| Thành phần | Ý nghĩa | Kích thước khái quát |
|---|---|---|
| `sequence_input` | Cửa sổ chuỗi đầu vào | `(B, T, F)` |
| `last_core_features` | Bước cuối của 6 đặc trưng lõi | `(B, 6)` |
| `returns_slice` | 4 đặc trưng cho ANFIS returns | `(B, 4)` |
| `indicator_slice` | 2 đặc trưng cho ANFIS indicators | `(B, 2)` |
| `fuzzy_raw_params` | 4 tham số thô do nhánh mờ dự đoán | `(B, 4)` |
| `sequence_residual_params` | 4 tham số hiệu chỉnh do nhánh chuỗi dự đoán | `(B, 4)` |
| `raw_target_params` | Tổng của hai nhánh | `(B, 4)` |
| `target_output` | 4 tham số sau khi bị chặn vào miền hợp lệ | `(B, 4)` |
| `predicted OHLC` | Giá Open/High/Low/Close đã dựng lại | `(B, 4)` |

## Ý nghĩa của 4 đầu ra cuối cùng

| Thành phần | Nghĩa |
|---|---|
| `close_ret` | Tỉ lệ biến động của Close kế tiếp so với `current_close` |
| `open_gap` | Tỉ lệ biến động của Open kế tiếp so với `current_close` |
| `high_buffer` | Mức High vượt lên trên `max(Open, Close)` trong không gian log |
| `low_buffer` | Mức Low đi xuống dưới `min(Open, Close)` trong không gian log |

## Công thức dựng lại OHLC

```text
pred_close = current_close * (1 + close_ret)
pred_open  = current_close * (1 + open_gap)
body_high  = max(pred_open, pred_close)
body_low   = min(pred_open, pred_close)
pred_high  = body_high * exp(high_buffer)
pred_low   = body_low * exp(-low_buffer)
```

Hệ quả trực tiếp của cách dựng này là:

- `pred_high >= max(pred_open, pred_close)`
- `pred_low <= min(pred_open, pred_close)`

Nghĩa là mô hình mới bảo đảm hình học OHLC hợp lệ ngay từ cấu trúc đầu ra, không cần thêm luật phạt bên ngoài.
