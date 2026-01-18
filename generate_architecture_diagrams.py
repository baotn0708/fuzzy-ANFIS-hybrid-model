#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh sơ đồ kiến trúc mô hình Feature-Group ANFIS + BiLSTM
Từ tổng quan đến chi tiết từng thành phần
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Thiết lập font tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

output_dir = 'architecture_diagrams'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. SƠ ĐỒ TỔNG QUAN - HIGH LEVEL ARCHITECTURE
# ============================================================
def draw_overview():
    """Sơ đồ tổng quan toàn bộ kiến trúc"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Kiến Trúc Feature-Group ANFIS + BiLSTM', 
            ha='center', fontsize=16, fontweight='bold')
    
    # INPUT LAYER
    input_box = FancyBboxPatch((5.5, 8), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 8.4, 'Input: (batch, 60, 6)', ha='center', fontweight='bold')
    ax.text(7, 8.15, '60 timesteps x 6 features', ha='center', fontsize=8)
    
    # FEATURE EXTRACTION
    ax.arrow(7, 8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black')
    
    feature_box = FancyBboxPatch((2, 6.8), 10, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='gray', facecolor='lightyellow', linewidth=1.5)
    ax.add_patch(feature_box)
    ax.text(7, 7.1, 'Trích xuất đặc trưng timestep cuối', ha='center', fontsize=9, style='italic')
    
    # SPLIT INTO GROUPS
    ax.arrow(4, 6.8, -0.3, -0.6, head_width=0.15, head_length=0.1, fc='blue')
    ax.arrow(7, 6.8, 0, -0.6, head_width=0.15, head_length=0.1, fc='green')
    ax.arrow(10, 6.8, 0.3, -0.6, head_width=0.15, head_length=0.1, fc='purple')
    
    # GROUP 1: RETURNS ANFIS
    returns_box = FancyBboxPatch((0.5, 4.5), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='blue', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(returns_box)
    ax.text(2, 5.7, 'ANFIS Returns', ha='center', fontweight='bold', color='blue')
    ax.text(2, 5.4, '4 features:', ha='center', fontsize=8)
    ax.text(2, 5.15, 'Close_ret, Open_ret,', ha='center', fontsize=7)
    ax.text(2, 4.95, 'High_ret, Low_ret', ha='center', fontsize=7)
    ax.text(2, 4.7, '2 MFs → 16 rules', ha='center', fontsize=8, style='italic')
    
    # GROUP 2: INDICATORS ANFIS
    indic_box = FancyBboxPatch((5.5, 4.5), 3, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(indic_box)
    ax.text(7, 5.7, 'ANFIS Indicators', ha='center', fontweight='bold', color='green')
    ax.text(7, 5.4, '2 features:', ha='center', fontsize=8)
    ax.text(7, 5.15, 'range_pct, gap', ha='center', fontsize=7)
    ax.text(7, 4.7, '2 MFs → 4 rules', ha='center', fontsize=8, style='italic')
    
    # GROUP 3: BiLSTM
    bilstm_box = FancyBboxPatch((10.5, 4.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='purple', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(bilstm_box)
    ax.text(12, 5.7, 'BiLSTM Path', ha='center', fontweight='bold', color='purple')
    ax.text(12, 5.3, 'Toàn bộ chuỗi', ha='center', fontsize=8)
    ax.text(12, 5.05, '(60 timesteps)', ha='center', fontsize=7)
    ax.text(12, 4.8, 'BiLSTM 64→32 units', ha='center', fontsize=7)
    
    # CONCATENATE
    ax.arrow(2, 4.5, 0.5, -0.8, head_width=0.15, head_length=0.1, fc='black')
    ax.arrow(7, 4.5, 0, -0.8, head_width=0.15, head_length=0.1, fc='black')
    ax.arrow(12, 4.5, -0.5, -0.8, head_width=0.15, head_length=0.1, fc='black')
    
    concat_box = FancyBboxPatch((5, 2.8), 4, 0.7,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(7, 3.15, 'Concatenate', ha='center', fontweight='bold')
    ax.text(7, 2.95, 'ANFIS_ret(8) + ANFIS_ind(4) + BiLSTM(64)', ha='center', fontsize=7)
    
    # DENSE LAYERS
    ax.arrow(7, 2.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black')
    
    dense_box = FancyBboxPatch((5.5, 1.5), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#FFE0B2', linewidth=1.5)
    ax.add_patch(dense_box)
    ax.text(7, 1.8, 'Dense Layers (32 → 4)', ha='center', fontweight='bold')
    
    # OUTPUT
    ax.arrow(7, 1.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='black')
    
    output_box = FancyBboxPatch((5.5, 0.5), 3, 0.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#FFCCBC', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 0.75, 'Output: 4 Returns Prediction', ha='center', fontweight='bold')
    
    # Legend
    ax.text(0.5, 0.3, 'Ưu điểm:', fontweight='bold', fontsize=9)
    ax.text(0.5, 0.1, '• Tránh bùng nổ luật (20 rules vs 64 rules)', fontsize=7)
    ax.text(4.5, 0.1, '• Mỗi ANFIS có ý nghĩa ngữ nghĩa', fontsize=7)
    ax.text(8.5, 0.1, '• Có thể giải thích được', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/1_overview.png")
    plt.close()


# ============================================================
# 2. CHI TIẾT ANFIS LAYER - 5 LAYERS
# ============================================================
def draw_anfis_detail():
    """Sơ đồ chi tiết 5 lớp của ANFIS"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Kiến Trúc ANFIS (5 Lớp) - Ví dụ 2 features, 2 MFs', 
            ha='center', fontsize=14, fontweight='bold')
    
    # LAYER 1: Fuzzification
    ax.text(1, 8.5, 'Layer 1: Fuzzification', fontweight='bold', fontsize=11)
    ax.text(1, 8.2, '(Gaussian MFs)', fontsize=9, style='italic')
    
    # Input nodes
    Circle((1.5, 7.5), 0.15, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(Circle((1.5, 7.5), 0.15, color='lightblue', ec='black', linewidth=2))
    ax.text(1.5, 7.5, 'x₁', ha='center', va='center', fontsize=9)
    
    ax.add_patch(Circle((1.5, 6.5), 0.15, color='lightblue', ec='black', linewidth=2))
    ax.text(1.5, 6.5, 'x₂', ha='center', va='center', fontsize=9)
    
    # MF nodes
    for i, y in enumerate([7.8, 7.2]):
        ax.add_patch(Circle((3, y), 0.12, color='#FFE082', ec='black', linewidth=1.5))
        ax.text(3, y, f'A{i+1}', ha='center', va='center', fontsize=8)
        ax.arrow(1.65, 7.5, 1.2, y-7.5, head_width=0.08, head_length=0.1, fc='gray', ec='gray', alpha=0.6)
    
    for i, y in enumerate([6.8, 6.2]):
        ax.add_patch(Circle((3, y), 0.12, color='#FFE082', ec='black', linewidth=1.5))
        ax.text(3, y, f'B{i+1}', ha='center', va='center', fontsize=8)
        ax.arrow(1.65, 6.5, 1.2, y-6.5, head_width=0.08, head_length=0.1, fc='gray', ec='gray', alpha=0.6)
    
    ax.text(3, 5.8, 'μ(x) = exp(-(x-c)²/2σ²)', ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # LAYER 2: Rule Firing
    ax.text(5, 8.5, 'Layer 2: Rule Firing', fontweight='bold', fontsize=11)
    ax.text(5, 8.2, '(Product T-norm)', fontsize=9, style='italic')
    
    rules_y = [7.8, 7.3, 6.8, 6.3]
    rule_labels = ['R1: A1∧B1', 'R2: A1∧B2', 'R3: A2∧B1', 'R4: A2∧B2']
    
    for i, (y, label) in enumerate(zip(rules_y, rule_labels)):
        ax.add_patch(Circle((5, y), 0.12, color='#FFCCBC', ec='black', linewidth=1.5))
        ax.text(5, y, f'w{i+1}', ha='center', va='center', fontsize=8)
        ax.text(5.8, y, label, fontsize=7, va='center')
        
        # Arrows from MFs to rules
        if i == 0:  # A1, B1
            ax.arrow(3.12, 7.8, 1.75, 0, head_width=0.06, head_length=0.08, fc='blue', alpha=0.4)
            ax.arrow(3.12, 6.8, 1.75, 1, head_width=0.06, head_length=0.08, fc='blue', alpha=0.4)
    
    # LAYER 3: Normalization
    ax.text(7.5, 8.5, 'Layer 3: Normalize', fontweight='bold', fontsize=11)
    ax.text(7.5, 8.2, '(w̄ᵢ = wᵢ/Σwᵢ)', fontsize=9, style='italic')
    
    for i, y in enumerate(rules_y):
        ax.add_patch(Circle((7.5, y), 0.12, color='#C5E1A5', ec='black', linewidth=1.5))
        ax.text(7.5, y, f'w̄{i+1}', ha='center', va='center', fontsize=7)
        ax.arrow(5.12, y, 2.25, 0, head_width=0.06, head_length=0.08, fc='green', alpha=0.4)
    
    # LAYER 4: TSK Consequents
    ax.text(10, 8.5, 'Layer 4: TSK', fontweight='bold', fontsize=11)
    ax.text(10, 8.2, '(fᵢ = pᵢ·x + rᵢ)', fontsize=9, style='italic')
    
    for i, y in enumerate(rules_y):
        ax.add_patch(Rectangle((9.5, y-0.15), 0.5, 0.3, 
                               color='#E1BEE7', ec='black', linewidth=1.5))
        ax.text(9.75, y, f'f{i+1}', ha='center', va='center', fontsize=8)
        ax.arrow(7.62, y, 1.75, 0, head_width=0.06, head_length=0.08, fc='purple', alpha=0.4)
    
    ax.text(10, 5.8, 'f₁ = p₁₁x₁ + p₁₂x₂ + r₁', ha='center', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='#E1BEE7', alpha=0.5))
    
    # LAYER 5: Defuzzification
    ax.text(12, 8.5, 'Layer 5: Output', fontweight='bold', fontsize=11)
    ax.text(12, 8.2, '(y = Σw̄ᵢfᵢ)', fontsize=9, style='italic')
    
    ax.add_patch(Circle((12, 7), 0.2, color='#FFAB91', ec='black', linewidth=2))
    ax.text(12, 7, 'OUT', ha='center', va='center', fontsize=9, fontweight='bold')
    
    for y in rules_y:
        ax.arrow(10, y, 1.7, 7-y, head_width=0.08, head_length=0.1, fc='red', alpha=0.3)
    
    # Math equations panel
    eq_box = FancyBboxPatch((0.5, 0.5), 13, 4,
                           boxstyle="round,pad=0.15",
                           edgecolor='navy', facecolor='#E8EAF6', linewidth=2, alpha=0.3)
    ax.add_patch(eq_box)
    
    ax.text(7, 4, 'Công Thức Toán Học', ha='center', fontweight='bold', fontsize=12)
    
    eqs = [
        "Layer 1: μₐᵢ(x) = exp(-(x-cᵢ)²/(2σᵢ²))  [Gaussian MF]",
        "Layer 2: wᵢ = Πⱼ μⱼ(xⱼ)  [Product T-norm]",
        "Layer 3: w̄ᵢ = wᵢ / Σₖwₖ  [Normalization]",
        "Layer 4: fᵢ = Σⱼ(pᵢⱼ·xⱼ) + rᵢ  [TSK first-order]",
        "Layer 5: y = Σᵢ(w̄ᵢ·fᵢ)  [Weighted average]"
    ]
    
    for i, eq in enumerate(eqs):
        ax.text(1, 3.3 - i*0.5, eq, fontsize=9, family='monospace')
    
    # Parameters info
    ax.text(10.5, 1.2, 'Tham số học:', fontweight='bold', fontsize=10)
    ax.text(10.5, 0.9, '• cᵢ, σᵢ (MF centers & widths)', fontsize=8)
    ax.text(10.5, 0.7, '• pᵢⱼ, rᵢ (Consequent params)', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_anfis_detail.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/2_anfis_detail.png")
    plt.close()


# ============================================================
# 3. FEATURE GROUPS - RETURNS vs INDICATORS
# ============================================================
def draw_feature_groups():
    """Sơ đồ phân nhóm features và logic ANFIS cho mỗi nhóm"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # GROUP 1: RETURNS ANFIS
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.5, 'ANFIS Returns (16 rules)', ha='center', fontsize=13, fontweight='bold', color='blue')
    
    # Input features
    features = ['Close_ret', 'Open_ret', 'High_ret', 'Low_ret']
    for i, feat in enumerate(features):
        y = 8 - i*0.6
        ax.add_patch(FancyBboxPatch((0.5, y-0.2), 2, 0.4,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='blue', facecolor='#E3F2FD', linewidth=1.5))
        ax.text(1.5, y, feat, ha='center', va='center', fontsize=9)
        
        # 2 MFs per feature
        for j in range(2):
            mf_y = y + (j-0.5)*0.3
            ax.add_patch(Circle((3.5, mf_y), 0.1, color='#FFE082', ec='blue'))
            ax.text(3.5, mf_y, f'{"L" if j==0 else "H"}', ha='center', va='center', fontsize=7)
            ax.arrow(2.5, y, 0.85, mf_y-y, head_width=0.08, head_length=0.08, fc='blue', alpha=0.3)
    
    # Rules visualization (sample 4 of 16)
    ax.text(5.5, 8.5, 'Ví dụ rules:', fontsize=10, fontweight='bold')
    sample_rules = [
        'R1: L∧L∧L∧L → Bear market',
        'R2: H∧H∧H∧H → Bull market',
        'R3: L∧H∧-∧- → Gap up',
        'R16: -∧-∧-∧H → ...'
    ]
    for i, rule in enumerate(sample_rules):
        ax.text(5.5, 8 - i*0.4, rule, fontsize=7.5, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.text(5, 1.5, '2⁴ = 16 fuzzy rules', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))
    ax.text(5, 0.8, 'Mỗi rule mô tả 1 chế độ thị trường', ha='center', fontsize=8, style='italic')
    
    # GROUP 2: INDICATORS ANFIS
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.5, 'ANFIS Indicators (4 rules)', ha='center', fontsize=13, fontweight='bold', color='green')
    
    # Input features
    features = ['range_pct', 'gap']
    for i, feat in enumerate(features):
        y = 8 - i*0.8
        ax.add_patch(FancyBboxPatch((0.5, y-0.25), 2.5, 0.5,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='green', facecolor='#E8F5E9', linewidth=1.5))
        ax.text(1.75, y, feat, ha='center', va='center', fontsize=10)
        
        # Description
        desc = '(High-Low)/Close' if feat == 'range_pct' else '(Open-Prev_Close)/Prev_Close'
        ax.text(1.75, y-0.15, desc, ha='center', fontsize=6, style='italic')
        
        # 2 MFs
        for j in range(2):
            mf_y = y + (j-0.5)*0.35
            ax.add_patch(Circle((4, mf_y), 0.12, color='#FFE082', ec='green'))
            ax.text(4, mf_y, f'{"LOW" if j==0 else "HIGH"}', ha='center', va='center', fontsize=7)
            ax.arrow(3, y, 0.85, mf_y-y, head_width=0.1, head_length=0.1, fc='green', alpha=0.3)
    
    # All 4 rules
    ax.text(5.5, 8.5, 'Tất cả 4 rules:', fontsize=10, fontweight='bold')
    all_rules = [
        'R1: range_LOW ∧ gap_LOW  → Low volatility, no gap',
        'R2: range_LOW ∧ gap_HIGH → Low vol, gap present',
        'R3: range_HIGH ∧ gap_LOW → High vol, no gap',
        'R4: range_HIGH ∧ gap_HIGH → High vol + gap'
    ]
    for i, rule in enumerate(all_rules):
        ax.text(5.5, 7.8 - i*0.5, rule, fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.text(5, 1.5, '2² = 4 fuzzy rules', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax.text(5, 0.8, 'Mô tả volatility và gap patterns', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_feature_groups.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/3_feature_groups.png")
    plt.close()


# ============================================================
# 4. BiLSTM COMPONENT
# ============================================================
def draw_bilstm():
    """Sơ đồ chi tiết BiLSTM path"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(6, 9.5, 'BiLSTM Temporal Processing', ha='center', fontsize=14, fontweight='bold')
    
    # Input sequence
    ax.text(6, 8.8, 'Input: (batch, 60, 6)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Timesteps visualization
    for i in range(5):
        x = 2 + i*1.5
        ax.add_patch(Rectangle((x, 7.8), 0.6, 0.5, 
                               edgecolor='blue', facecolor='#BBDEFB', linewidth=1))
        ax.text(x+0.3, 8.05, f't-{60-i}' if i < 4 else '...', ha='center', fontsize=7)
    
    ax.add_patch(Rectangle((9, 7.8), 0.6, 0.5,
                           edgecolor='red', facecolor='#FFCDD2', linewidth=2))
    ax.text(9.3, 8.05, 't', ha='center', fontsize=7, fontweight='bold')
    
    # BiLSTM Layer 1
    ax.arrow(6, 7.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black')
    
    lstm1_box = FancyBboxPatch((3, 6.5), 6, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(lstm1_box)
    ax.text(6, 7, 'Bidirectional LSTM (64 units)', ha='center', fontweight='bold')
    ax.text(6, 6.75, 'return_sequences=True', ha='center', fontsize=8, style='italic')
    
    # Forward/Backward arrows
    ax.arrow(3.5, 6.6, 4.5, 0, head_width=0.08, head_length=0.15, fc='blue', alpha=0.4, linewidth=2)
    ax.text(5.5, 6.45, 'Forward', ha='center', fontsize=7, color='blue')
    ax.arrow(8, 6.9, -4.5, 0, head_width=0.08, head_length=0.15, fc='red', alpha=0.4, linewidth=2)
    ax.text(5.5, 7.05, 'Backward', ha='center', fontsize=7, color='red')
    
    # Dropout
    ax.arrow(6, 6.5, 0, -0.4, head_width=0.15, head_length=0.08, fc='gray')
    dropout_box = FancyBboxPatch((5, 5.7), 2, 0.3,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='gray', facecolor='#EEEEEE', linewidth=1, linestyle='dashed')
    ax.add_patch(dropout_box)
    ax.text(6, 5.85, 'Dropout (0.2)', ha='center', fontsize=8)
    
    # BiLSTM Layer 2
    ax.arrow(6, 5.7, 0, -0.5, head_width=0.15, head_length=0.08, fc='black')
    
    lstm2_box = FancyBboxPatch((3.5, 4.4), 5, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(lstm2_box)
    ax.text(6, 4.95, 'Bidirectional LSTM (32 units)', ha='center', fontweight='bold')
    ax.text(6, 4.7, 'return_sequences=False', ha='center', fontsize=8, style='italic')
    ax.text(6, 4.5, '→ Chỉ lấy output cuối', ha='center', fontsize=7, color='red')
    
    # Output
    ax.arrow(6, 4.4, 0, -0.4, head_width=0.15, head_length=0.08, fc='gray')
    dropout2_box = FancyBboxPatch((5, 3.6), 2, 0.3,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='gray', facecolor='#EEEEEE', linewidth=1, linestyle='dashed')
    ax.add_patch(dropout2_box)
    ax.text(6, 3.75, 'Dropout (0.2)', ha='center', fontsize=8)
    
    ax.arrow(6, 3.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    
    output_box = FancyBboxPatch((4.5, 2.7), 3, 0.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#FFCCBC', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 2.95, 'Output: (batch, 64)', ha='center', fontweight='bold')
    ax.text(6, 2.8, 'Temporal features', ha='center', fontsize=8, style='italic')
    
    # LSTM Cell detail
    cell_box = FancyBboxPatch((0.5, 0.5), 11, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='navy', facecolor='#E8EAF6', linewidth=1.5, alpha=0.4)
    ax.add_patch(cell_box)
    
    ax.text(6, 2.1, 'LSTM Cell (simplified):', fontweight='bold', fontsize=10)
    gates = [
        'Forget gate:  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)',
        'Input gate:   iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)',
        'Output gate:  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)',
        'Cell state:   cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ] + bc)',
        'Hidden:       hₜ = oₜ⊙tanh(cₜ)'
    ]
    for i, gate in enumerate(gates):
        ax.text(1, 1.7 - i*0.25, gate, fontsize=7.5, family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_bilstm_detail.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/4_bilstm_detail.png")
    plt.close()


# ============================================================
# 5. DATA FLOW - END TO END
# ============================================================
def draw_data_flow():
    """Sơ đồ luồng dữ liệu từ đầu đến cuối"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    ax.text(7, 10.5, 'Data Flow: Raw Data → Prediction', ha='center', 
            fontsize=15, fontweight='bold')
    
    # Raw data
    y = 9.5
    ax.add_patch(FancyBboxPatch((2, y-0.3), 10, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#BBDEFB', linewidth=2))
    ax.text(7, y, 'Raw OHLC Data: Close, Open, High, Low, Volume', ha='center', fontweight='bold')
    
    # Feature engineering
    ax.arrow(7, y-0.3, 0, -0.4, head_width=0.2, head_length=0.1, fc='black')
    y -= 1.2
    ax.add_patch(FancyBboxPatch((1.5, y-0.4), 11, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='#C8E6C9', linewidth=2))
    ax.text(7, y+0.15, 'Feature Engineering', ha='center', fontweight='bold', fontsize=11)
    ax.text(7, y-0.05, '• Returns: Close_ret, Open_ret, High_ret, Low_ret', ha='center', fontsize=8)
    ax.text(7, y-0.25, '• Indicators: range_pct = (H-L)/C, gap = (O-Prev_C)/Prev_C', ha='center', fontsize=8)
    
    # Standardization
    ax.arrow(7, y-0.4, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    y -= 0.9
    ax.add_patch(FancyBboxPatch((4, y-0.2), 6, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='blue', facecolor='#E3F2FD', linewidth=1.5))
    ax.text(7, y, 'StandardScaler: (x - μ) / σ', ha='center', fontsize=9)
    
    # Sequence creation
    ax.arrow(7, y-0.2, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    y -= 0.9
    ax.add_patch(FancyBboxPatch((3, y-0.3), 8, 0.6,
                                boxstyle="round,pad=0.08",
                                edgecolor='purple', facecolor='#F3E5F5', linewidth=2))
    ax.text(7, y+0.05, 'Create Sequences: look_back=60', ha='center', fontweight='bold', fontsize=10)
    ax.text(7, y-0.15, 'Shape: (batch, 60, 6) = (samples, timesteps, features)', ha='center', fontsize=8)
    
    # Model input
    ax.arrow(7, y-0.3, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', linewidth=2)
    y -= 1.2
    ax.add_patch(FancyBboxPatch((2.5, y-0.5), 9, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='#FFEBEE', linewidth=3))
    ax.text(7, y+0.25, 'MODEL PROCESSING', ha='center', fontweight='bold', fontsize=12, color='red')
    
    # Three parallel paths
    ax.arrow(4, y-0.5, -0.5, -0.8, head_width=0.12, head_length=0.08, fc='blue', linewidth=1.5)
    ax.arrow(7, y-0.5, 0, -0.8, head_width=0.12, head_length=0.08, fc='green', linewidth=1.5)
    ax.arrow(10, y-0.5, 0.5, -0.8, head_width=0.12, head_length=0.08, fc='purple', linewidth=1.5)
    
    y -= 2
    # Path 1: Returns ANFIS
    ax.add_patch(FancyBboxPatch((0.5, y-0.3), 3, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='blue', facecolor='#E3F2FD'))
    ax.text(2, y, 'ANFIS_ret', ha='center', fontsize=9, fontweight='bold')
    ax.text(2, y-0.15, '16 rules → 8D', ha='center', fontsize=7)
    
    # Path 2: Indicators ANFIS
    ax.add_patch(FancyBboxPatch((5.5, y-0.3), 3, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='green', facecolor='#E8F5E9'))
    ax.text(7, y, 'ANFIS_ind', ha='center', fontsize=9, fontweight='bold')
    ax.text(7, y-0.15, '4 rules → 4D', ha='center', fontsize=7)
    
    # Path 3: BiLSTM
    ax.add_patch(FancyBboxPatch((10.5, y-0.3), 3, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='purple', facecolor='#F3E5F5'))
    ax.text(12, y, 'BiLSTM', ha='center', fontsize=9, fontweight='bold')
    ax.text(12, y-0.15, '64→32 → 64D', ha='center', fontsize=7)
    
    # Concatenation
    ax.arrow(2, y-0.3, 3, -0.5, head_width=0.1, head_length=0.08, fc='black')
    ax.arrow(7, y-0.3, 0, -0.5, head_width=0.1, head_length=0.08, fc='black')
    ax.arrow(12, y-0.3, -3, -0.5, head_width=0.1, head_length=0.08, fc='black')
    
    y -= 1.2
    ax.add_patch(FancyBboxPatch((5, y-0.25), 4, 0.5,
                                boxstyle="round,pad=0.08",
                                edgecolor='black', facecolor='#FFF9C4', linewidth=2))
    ax.text(7, y, 'Concatenate: 8 + 4 + 64 = 76D', ha='center', fontweight='bold', fontsize=10)
    
    # Dense layers
    ax.arrow(7, y-0.25, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    y -= 0.8
    ax.add_patch(FancyBboxPatch((5.5, y-0.2), 3, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='orange', facecolor='#FFE0B2'))
    ax.text(7, y, 'Dense(32) + ReLU + Dropout', ha='center', fontsize=9)
    
    ax.arrow(7, y-0.2, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    y -= 0.7
    ax.add_patch(FancyBboxPatch((5.5, y-0.2), 3, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='brown', facecolor='#FFCCBC'))
    ax.text(7, y, 'Dense(4) → Output', ha='center', fontsize=9, fontweight='bold')
    
    # Prediction output
    ax.arrow(7, y-0.2, 0, -0.4, head_width=0.2, head_length=0.1, fc='red', linewidth=2)
    y -= 0.9
    ax.add_patch(FancyBboxPatch((4, y-0.3), 6, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='#FFCDD2', linewidth=2.5))
    ax.text(7, y+0.05, 'Predicted Returns (4D)', ha='center', fontweight='bold', fontsize=11)
    ax.text(7, y-0.15, '[Close_ret, Open_ret, High_ret, Low_ret]', ha='center', fontsize=8, style='italic')
    
    # Conversion to prices
    ax.arrow(7, y-0.3, 0, -0.3, head_width=0.15, head_length=0.08, fc='black')
    y -= 0.8
    ax.add_patch(FancyBboxPatch((3.5, y-0.3), 7, 0.6,
                                boxstyle="round,pad=0.08",
                                edgecolor='green', facecolor='#C8E6C9', linewidth=2))
    ax.text(7, y+0.05, 'Convert to Prices', ha='center', fontweight='bold', fontsize=10)
    ax.text(7, y-0.15, 'next_price = current_close × (1 + predicted_return)', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_data_flow.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/5_data_flow.png")
    plt.close()


# ============================================================
# 6. TRAINING PROCESS
# ============================================================
def draw_training_process():
    """Sơ đồ quy trình training"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(6, 9.5, 'Training Process Flow', ha='center', fontsize=14, fontweight='bold')
    
    steps = [
        ('Load Data', 'CSV → DataFrame', '#BBDEFB', 8.5),
        ('Feature Engineering', 'Calculate returns & indicators', '#C8E6C9', 7.5),
        ('Standardization', 'StandardScaler fit_transform', '#E3F2FD', 6.5),
        ('K-Means Init', 'Initialize MF centers (2 clusters)', '#FFF9C4', 5.5),
        ('Create Model', 'Build ANFIS + BiLSTM', '#F3E5F5', 4.5),
        ('Train (150 epochs)', 'MSE loss, Adam optimizer', '#FFCCBC', 3.5),
        ('Early Stopping', 'Patience=20, monitor val_loss', '#FFE0B2', 2.5),
        ('Evaluate', 'Calculate R², MAPE, DA', '#C8E6C9', 1.5),
        ('Extract Rules', 'Get interpretable fuzzy rules', '#E1BEE7', 0.5),
    ]
    
    for i, (title, desc, color, y) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((2, y-0.35), 8, 0.7,
                                    boxstyle="round,pad=0.08",
                                    edgecolor='black', facecolor=color, linewidth=1.5))
        ax.text(6, y+0.1, f'{i+1}. {title}', ha='center', fontweight='bold', fontsize=10)
        ax.text(6, y-0.15, desc, ha='center', fontsize=8, style='italic')
        
        if i < len(steps) - 1:
            ax.arrow(6, y-0.35, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_training_process.png', dpi=300, bbox_inches='tight')
    print(f"✅ Đã tạo: {output_dir}/6_training_process.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎨 SINH SƠ ĐỒ KIẾN TRÚC MÔ HÌNH FEATURE-GROUP ANFIS + BiLSTM")
    print("="*60 + "\n")
    
    draw_overview()
    draw_anfis_detail()
    draw_feature_groups()
    draw_bilstm()
    draw_data_flow()
    draw_training_process()
    
    print(f"\n{'='*60}")
    print(f"✅ ĐÃ TẠO 6 SƠ ĐỒ TRONG THƯ MỤC: {output_dir}/")
    print(f"{'='*60}")
    print("\nDanh sách sơ đồ:")
    print("  1. 1_overview.png          - Tổng quan kiến trúc")
    print("  2. 2_anfis_detail.png      - Chi tiết 5 lớp ANFIS")
    print("  3. 3_feature_groups.png    - Phân nhóm features")
    print("  4. 4_bilstm_detail.png     - Chi tiết BiLSTM")
    print("  5. 5_data_flow.png         - Luồng dữ liệu end-to-end")
    print("  6. 6_training_process.png  - Quy trình training")
    print("\n✨ HOÀN THÀNH!\n")
