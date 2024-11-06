import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
boston_housing_df = pd.read_csv(
    'boston-housing-reduced.csv')  # Update with your file path

# Define features (RM, RAD, INDUS, TAX) and target (MEDV)
X = boston_housing_df[['RM', 'RAD', 'INDUS', 'TAX']]
y = boston_housing_df['MEDV']

# Remove outliers based on Z-score (new step)
z_scores = np.abs((X - X.mean()) / X.std())
X = X[(z_scores < 3).all(axis=1)]
y = y.loc[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with Grid Search for RandomForestRegressor (removed 'auto' for max_features)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    # Removed 'auto' to avoid compatibility issues
    'max_features': ['sqrt', 'log2']
}
rf_regressor = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params = grid_search.best_params_
best_rf_regressor = grid_search.best_estimator_
y_best_rf_pred = best_rf_regressor.predict(X_test)

# Evaluate the best Random Forest model performance
best_rf_mse = mean_squared_error(y_test, y_best_rf_pred)
best_rf_r2 = r2_score(y_test, y_best_rf_pred)
print("Best Parameters for Random Forest:", best_params)
print("Best Random Forest - MSE:", best_rf_mse)
print("Best Random Forest - R²:", best_rf_r2)

# Cross-validation with the best Random Forest model (20-fold Stratified K-Fold)
stratified_cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
scores = cross_val_score(best_rf_regressor, X, y,
                         cv=stratified_cv, scoring='r2')
print("Cross-Validation R² Scores (20-fold Stratified):", scores)
print("Average Cross-Validation R² Score (20-fold Stratified):", scores.mean())

# Feature scaling for potential further modeling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split scaled data for further modeling
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate the best Random Forest model on scaled data
best_rf_regressor.fit(X_train_scaled, y_train_scaled)
y_scaled_rf_pred = best_rf_regressor.predict(X_test_scaled)
scaled_rf_mse = mean_squared_error(y_test_scaled, y_scaled_rf_pred)
scaled_rf_r2 = r2_score(y_test_scaled, y_scaled_rf_pred)
print("Best Random Forest on Scaled Data - MSE:", scaled_rf_mse)
print("Best Random Forest on Scaled Data - R²:", scaled_rf_r2)
