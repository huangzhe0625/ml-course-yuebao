#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习课程设计 - 余额宝资金流入流出预测
使用LightGBM + 时间序列特征 + 用户画像特征，改进版本
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("余额宝资金流入流出预测 - LightGBM模型(改进版)")
print("=" * 50)

# ========== 1. 读取数据 ==========
print("\n[1/6] 正在读取数据...")

# 用户申购赎回数据（主数据）
train_balance = pd.read_csv('E:/机器学习课程设计/余额宝资金预测_课程设计/data/user_balance_table.csv')

# 用户画像数据
user_profile = pd.read_csv('E:/机器学习课程设计/余额宝资金预测_课程设计/data/user_profile_table.csv')

# 收益率表
interest = pd.read_csv('E:/机器学习课程设计/余额宝资金预测_课程设计/data/mfd_day_share_interest.csv')

# 银行间拆借利率
shibor = pd.read_csv('E:/机器学习课程设计/余额宝资金预测_课程设计/data/mfd_bank_shibor.csv')

print(f"  申购赎回数据: {train_balance.shape}")
print(f"  用户画像数据: {user_profile.shape}")
print(f"  收益率数据: {interest.shape}")
print(f"  拆借利率数据: {shibor.shape}")

# ========== 2. 数据预处理 ==========
print("\n[2/6] 正在处理数据...")

# 转换日期格式
train_balance['report_date'] = pd.to_datetime(train_balance['report_date'], format='%Y%m%d')

# 合并用户画像
train_balance = train_balance.merge(user_profile, on='user_id', how='left')

# 处理缺失值（用户画像中可能有新用户没有信息）
train_balance['sex'] = train_balance['sex'].fillna(-1).astype(int)
train_balance['city'] = train_balance['city'].fillna(-1).astype(int)
train_balance['constellation'] = train_balance['constellation'].fillna('未知')

# 收益率和拆借利率日期处理
interest['mfd_date'] = pd.to_datetime(interest['mfd_date'], format='%Y%m%d')
shibor['mfd_date'] = pd.to_datetime(shibor['mfd_date'], format='%Y%m%d')

# 按日期汇总每日总申购和总赎回（包含用户画像分组信息）
daily = train_balance.groupby('report_date').agg({
    'total_purchase_amt': 'sum',
    'total_redeem_amt': 'sum',
    'sex': 'mean',  # 平均性别比例
    'city': 'nunique'  # 活跃城市数
}).reset_index()

daily.columns = ['report_date', 'purchase', 'redeem', 'avg_sex', 'active_cities']

print(f"  每日汇总数据: {daily.shape}")
print(f"  数据时间范围: {daily['report_date'].min()} ~ {daily['report_date'].max()}")

# ========== 3. 合并外部数据 ==========
print("\n[3/6] 正在合并外部数据...")

# 合并收益率
daily['date_int'] = daily['report_date'].dt.strftime('%Y%m%d').astype(int)
interest['date_int'] = interest['mfd_date'].dt.strftime('%Y%m%d').astype(int)
daily = daily.merge(interest[['date_int', 'mfd_daily_yield', 'mfd_7daily_yield']], 
                     on='date_int', how='left')

# 合并拆借利率
shibor['date_int'] = shibor['mfd_date'].dt.strftime('%Y%m%d').astype(int)
daily = daily.merge(shibor[['date_int', 'Interest_O_N']], 
                     on='date_int', how='left')

# 填充缺失值
daily = daily.ffill()
daily = daily.bfill()

print(f"  合并后数据: {daily.shape}")

# ========== 4. 构造时间序列特征 ==========
print("\n[4/6] 正在构造特征...")

# 时间特征
daily['year'] = daily['report_date'].dt.year
daily['month'] = daily['report_date'].dt.month
daily['day'] = daily['report_date'].dt.day
daily['weekday'] = daily['report_date'].dt.weekday
daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)
daily['is_month_start'] = (daily['day'] <= 5).astype(int)
daily['is_month_end'] = (daily['day'] >= 25).astype(int)

# 节假日特征（2014年9月中秋节和国庆节）
holiday_dates = ['2014-09-06', '2014-09-07', '2014-09-08',  # 中秋节
                 '2014-09-28', '2014-09-29', '2014-09-30']  # 国庆节前
daily['is_holiday'] = daily['report_date'].isin(holiday_dates).astype(int)
daily['is_pre_holiday'] = daily['report_date'].isin(['2014-09-05', '2014-09-27']).astype(int)

# 滞后特征（更多天数）
for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21]:
    daily[f'purchase_lag{lag}'] = daily['purchase'].shift(lag)
    daily[f'redeem_lag{lag}'] = daily['redeem'].shift(lag)

# 移动平均特征
for window in [3, 7, 14, 21]:
    daily[f'purchase_ma{window}'] = daily['purchase'].shift(1).rolling(window=window).mean()
    daily[f'redeem_ma{window}'] = daily['redeem'].shift(1).rolling(window=window).mean()

# 同比特征（去年同期）
daily['purchase_yoy'] = daily['purchase'].shift(365)
daily['redeem_yoy'] = daily['redeem'].shift(365)

# 删除有缺失值的行
daily = daily.dropna()

print(f"  特征构造后: {daily.shape}")

# ========== 5. 训练模型 ==========
print("\n[5/6] 正在训练模型...")

# 特征列
feature_cols = [c for c in daily.columns if c not in ['report_date', 'purchase', 'redeem', 'date_int']]

X = daily[feature_cols]
y_purchase = daily['purchase']
y_redeem = daily['redeem']

# 划分训练集和验证集
split_idx = int(len(daily) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_pur_train, y_pur_val = y_purchase.iloc[:split_idx], y_purchase.iloc[split_idx:]
y_red_train, y_red_val = y_redeem.iloc[:split_idx], y_redeem.iloc[split_idx:]

print(f"  训练集: {len(X_train)} 天, 验证集: {len(X_val)} 天")

# LightGBM参数（优化后）
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# 训练申购预测模型
print("  训练申购模型...")
train_data_pur = lgb.Dataset(X_train, label=y_pur_train)
val_data_pur = lgb.Dataset(X_val, label=y_pur_val, reference=train_data_pur)
model_purchase = lgb.train(
    params, train_data_pur, num_boost_round=5000,
    valid_sets=[train_data_pur, val_data_pur],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
)

# 训练赎回预测模型
print("  训练赎回模型...")
train_data_red = lgb.Dataset(X_train, label=y_red_train)
val_data_red = lgb.Dataset(X_val, label=y_red_val, reference=train_data_red)
model_redeem = lgb.train(
    params, train_data_red, num_boost_round=5000,
    valid_sets=[train_data_red, val_data_red],
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
)

# 评估
pur_pred = model_purchase.predict(X_val, num_iteration=model_purchase.best_iteration)
red_pred = model_redeem.predict(X_val, num_iteration=model_redeem.best_iteration)

pur_mae = mean_absolute_error(y_pur_val, pur_pred)
red_mae = mean_absolute_error(y_red_val, red_pred)

print(f"\n  验证集MAE - 申购: {pur_mae:.2f}, 赎回: {red_mae:.2f}")

# 特征重要性
print("\nTop 10 申购重要特征:")
feature_importance_pur = pd.DataFrame({
    'feature': X.columns,
    'importance': model_purchase.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(feature_importance_pur.head(10).to_string(index=False))

print("\nTop 10 赎回重要特征:")
feature_importance_red = pd.DataFrame({
    'feature': X.columns,
    'importance': model_redeem.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(feature_importance_red.head(10).to_string(index=False))

# ========== 6. 预测2014年9月并生成提交文件 ==========
print("\n[6/6] 正在预测2014年9月并生成提交文件...")

pred_dates = pd.date_range('2014-09-01', '2014-09-30', freq='D')
predictions = []

for date in pred_dates:
    row = daily.iloc[-1:].copy()
    row['report_date'] = date
    row['year'] = date.year
    row['month'] = date.month
    row['day'] = date.day
    row['weekday'] = date.weekday()
    row['is_weekend'] = int(date.weekday() >= 5)
    row['is_month_start'] = int(date.day <= 5)
    row['is_month_end'] = int(date.day >= 25)
    row['is_holiday'] = int(date.strftime('%Y-%m-%d') in holiday_dates)
    row['is_pre_holiday'] = int(date.strftime('%Y-%m-%d') in ['2014-09-05', '2014-09-27'])

    # lag特征
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21]:
        row[f'purchase_lag{lag}'] = daily['purchase'].iloc[-lag]
        row[f'redeem_lag{lag}'] = daily['redeem'].iloc[-lag]

    # 移动平均
    for window in [3, 7, 14, 21]:
        row[f'purchase_ma{window}'] = daily['purchase'].iloc[-window:].mean()
        row[f'redeem_ma{window}'] = daily['redeem'].iloc[-window:].mean()

    # 同比
    row['purchase_yoy'] = daily['purchase'].iloc[-365] if len(daily) >= 365 else daily['purchase'].iloc[-1]
    row['redeem_yoy'] = daily['redeem'].iloc[-365] if len(daily) >= 365 else daily['redeem'].iloc[-1]

    X_pred = row[feature_cols]
    pur = model_purchase.predict(X_pred)[0]
    red = model_redeem.predict(X_pred)[0]

    predictions.append({
        'report_date': int(date.strftime('%Y%m%d')),
        'purchase': max(0, int(pur)),
        'redeem': max(0, int(red))
    })

    new_row = row.copy()
    new_row['purchase'] = pur
    new_row['redeem'] = red
    daily = pd.concat([daily, new_row], ignore_index=True)

# 保存结果
result = pd.DataFrame(predictions)
result.to_csv('E:/机器学习课程设计/余额宝资金预测_课程设计/tc_comp_predict_table.csv', index=False, header=False)

print("\n" + "=" * 50)
print("✅ 预测完成！")
print("=" * 50)
print(f"\n提交文件: E:/机器学习课程设计/余额宝资金预测_课程设计/tc_comp_predict_table.csv")
print(f"预测时间范围: 2014-09-01 ~ 2014-09-30")
print(f"\n预测结果预览:")
print(result.head(10).to_string(index=False))
print(f"\n...")
print(result.tail(5).to_string(index=False))

print(f"\n申购总额范围: [{result['purchase'].min():,}, {result['purchase'].max():,}]")
print(f"赎回总额范围: [{result['redeem'].min():,}, {result['redeem'].max():,}]")
print(f"\n请将 tc_comp_predict_table.csv 提交到天池竞赛平台")
print("=" * 50)
