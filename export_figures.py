#!/usr/bin/env python3
"""Export figures from the notebook analysis for the paper."""

import sys
sys.path.insert(0, '/home/michaeldejournett/Intro-To-Machine-Learning/project')

import pickle
import json
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_dir = Path('/home/michaeldejournett/Intro-To-Machine-Learning/project')
paper_fig_dir = Path('/home/michaeldejournett/Intro-To-Machine-Learning/paper/figures')
paper_fig_dir.mkdir(parents=True, exist_ok=True)

# Load data
zip_path = Path('/home/michaeldejournett/Intro-To-Machine-Learning/archive.zip')
with zipfile.ZipFile(zip_path, 'r') as z:
    rides = pd.read_csv(z.open('cab_rides.csv'))
    weather = pd.read_csv(z.open('weather.csv'))

# Merge and sample
merged = rides.merge(weather, on=['day', 'month', 'year'], how='left')
df = merged.sample(n=20000, random_state=42).reset_index(drop=True)

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Separate features and target
y = df['price']
X = df.drop(['price', 'surge_multiplier'], axis=1)

# Time features
X['hour'] = pd.to_datetime(X['time_stamp']).dt.hour
X['day_of_week'] = pd.to_datetime(X['date']).dt.dayofweek
X['month'] = pd.to_datetime(X['date']).dt.month

# Drop columns that aren't needed
X = X.drop(['date', 'time_stamp', 'day', 'year'], axis=1, errors='ignore')

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# Label encode categorical columns
cat_cols = ['cab_type', 'name', 'source', 'destination']
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col + '_enc'] = le.fit_transform(X[col])
        X = X.drop(col, axis=1)

# Ensure features are numeric
features = [col for col in X.columns if col in ['distance', 'hour', 'day_of_week', 'month', 
                                                   'temp', 'clouds', 'pressure', 'rain', 
                                                   'humidity', 'wind', 'cab_type_enc', 
                                                   'name_enc', 'source_enc', 'destination_enc']]
X = X[features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train models
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
preds_lr = lr.predict(X_test_sc)

rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)

print("Models trained successfully")

# Export Actual vs Predicted scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, preds_lr, alpha=0.2, s=5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)', fontsize=11)
axes[0].set_ylabel('Predicted Price ($)', fontsize=11)
axes[0].set_title('Linear Regression — Actual vs Predicted', fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].scatter(y_test, preds_rf, alpha=0.2, s=5, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price ($)', fontsize=11)
axes[1].set_ylabel('Predicted Price ($)', fontsize=11)
axes[1].set_title('Random Forest — Actual vs Predicted', fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
actual_vs_pred_path = paper_fig_dir / 'actual_vs_predicted.png'
plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
print(f"Saved: {actual_vs_pred_path}")
plt.close()

# Export Feature importance plots
importances_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
importances_lr = pd.Series(np.abs(lr.coef_), index=features).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

importances_lr.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Linear Regression — Feature Coefficients (abs)', fontsize=12)
axes[0].set_xlabel('Coefficients (abs)', fontsize=11)

importances_rf.plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('Random Forest — Feature Importances', fontsize=12)
axes[1].set_xlabel('Importance', fontsize=11)

plt.tight_layout()
feature_importance_path = paper_fig_dir / 'feature_importance.png'
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
print(f"Saved: {feature_importance_path}")
plt.close()

print("\nAll figures exported successfully!")
