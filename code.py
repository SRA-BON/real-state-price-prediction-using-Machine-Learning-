# -*- coding: utf-8 -*-
"""
REFACTORED CODE for Melbourne Housing Price Prediction
- Removed Colab dependencies
- Fixed Data Leakage (Imputation and Scaling done after split)
- Linearized execution flow to prevent NameErrors
- Standardized model evaluation
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# --- DATA LOADING ---
file_path = 'datasets/Melbourne_housing_FULL.csv'
df = pd.read_csv(file_path)

print(f"Initial Dataset Shape: {df.shape}")

# --- INITIAL CLEANING ---
df_copy = df.copy()
df_copy.dropna(subset=['Price'], inplace=True)

# --- FEATURE ENGINEERING (Non-Leakage) ---
df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')
df_copy['Month_Sold'] = df_copy['Date'].dt.month
df_copy['Year_Sold'] = df_copy['Date'].dt.year

# Handle high cardinality features by grouping rare categories
def group_rare_categories(df, col, threshold=50):
    counts = df[col].value_counts()
    rare_cats = counts[counts < threshold].index
    df[col] = df[col].apply(lambda x: 'Other' if x in rare_cats else x)
    return df

for col in ['Suburb', 'SellerG']:
    df_copy = group_rare_categories(df_copy, col)

# Dropping redundant or purely unique columns
columns_to_drop = [
    'Address', 'Bedroom2', 'Postcode', 'CouncilArea', 
    'Lattitude', 'Longtitude', 'Date'
]
df_clean = df_copy.drop(columns=columns_to_drop)

# --- DATA SPLIT ---
X = df_clean.drop(columns=['Price'])
y = df_clean['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PREPROCESSING ---

def preprocess_data(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    
    # 1. Impute YearBuilt and Year_Sold to calculate Age_at_Sale
    year_built_median = train['YearBuilt'].median()
    train['YearBuilt'] = train['YearBuilt'].fillna(year_built_median)
    test['YearBuilt'] = test['YearBuilt'].fillna(year_built_median)
    
    year_sold_median = train['Year_Sold'].median()
    train['Year_Sold'] = train['Year_Sold'].fillna(year_sold_median)
    test['Year_Sold'] = test['Year_Sold'].fillna(year_sold_median)
    
    train['Age_at_Sale'] = (train['Year_Sold'] - train['YearBuilt']).clip(lower=0)
    test['Age_at_Sale'] = (test['Year_Sold'] - test['YearBuilt']).clip(lower=0)
    
    train.drop(columns=['YearBuilt', 'Year_Sold'], inplace=True)
    test.drop(columns=['YearBuilt', 'Year_Sold'], inplace=True)

    # 2. Impute other numericals
    num_cols = ['Rooms', 'Bathroom', 'Distance', 'Landsize', 'BuildingArea', 'Propertycount', 'Month_Sold']
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    # 3. Create Total_Rooms
    train['Total_Rooms'] = train['Rooms'] + train['Bathroom']
    test['Total_Rooms'] = test['Rooms'] + test['Bathroom']
    train.drop(columns=['Rooms', 'Bathroom'], inplace=True)
    test.drop(columns=['Rooms', 'Bathroom'], inplace=True)
    
    # 4. Outlier Capping
    outlier_cols = ['Landsize', 'BuildingArea']
    for col in outlier_cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
        test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)
        
    # 5. Categorical Encoding (One-Hot)
    cat_cols = ['Type', 'Method', 'Regionname', 'Suburb', 'SellerG']
    for col in cat_cols:
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)
    
    train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
    test = pd.get_dummies(test, columns=cat_cols, drop_first=True)
    
    # Align columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)
    
    return train, test

X_train_pre, X_test_pre = preprocess_data(X_train, X_test)

# --- FEATURE SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pre)
X_test_scaled = scaler.transform(X_test_pre)

# --- TARGET TRANSFORMATION ---
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# --- MODEL TRAINING & EVALUATION ---
print("\n--- Model Training & Cross-Validation ---")

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Lasso": Lasso(alpha=0.01, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42)
}

# --- HYPERPARAMETER TUNING (Example for Random Forest) ---
print("\n--- Tuning Random Forest ---")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_tuned = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=5, cv=3, scoring='r2', random_state=42, n_jobs=-1
)
rf_tuned.fit(X_train_pre, y_train_log)
print(f"Best RF Params: {rf_tuned.best_params_}")

# Add tuned model to comparison
models["Random Forest (Tuned)"] = rf_tuned.best_estimator_

results = []

for name, model in models.items():
    # Use scaled data for Linear models, original for Tree models
    curr_X_train = X_train_scaled if name in ["Lasso", "Ridge"] else X_train_pre.values
    curr_X_test = X_test_scaled if name in ["Lasso", "Ridge"] else X_test_pre.values
    
    # Cross-Validation (on Training Set)
    cv_scores = cross_val_score(model, curr_X_train, y_train_log, cv=5, scoring='r2')
    
    # Train on Log Target
    model.fit(curr_X_train, y_train_log)
    
    # Predict and transform back
    y_pred_log = model.predict(curr_X_test)
    y_pred = np.expm1(y_pred_log)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name, 
        "MAE": mae, 
        "R2 Score": r2, 
        "CV R2 (Mean)": cv_scores.mean()
    })
    print(f"{name}: MAE = {mae:,.2f}, R2 = {r2:.4f}, CV R2 = {cv_scores.mean():.4f}")

# --- FINAL COMPARISON ---
# FIXED: Sorted results by R2 score for easier analysis
comparison_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
print("\nFinal Model Performance Comparison:")
print(comparison_df)

# Visualizing results
plt.figure(figsize=(10, 6))
sns.barplot(x='R2 Score', y='Model', data=comparison_df, palette='viridis', hue='Model', legend=False)
plt.title('Model Comparison - R2 Score')
plt.show()