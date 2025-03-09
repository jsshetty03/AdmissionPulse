import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

# Load the dataset with flexible column name handling
def load_and_prepare_data(file_path="/Users/jashanshetty29/newp/1000entries.csv"):
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None, None
        
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns")
        print(f"Columns found: {', '.join(dataset.columns)}")
        
        target_col = None
        for col in dataset.columns:
            if 'admit' in col.lower() or 'chance' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            print("Could not automatically identify the admission chance column.")
            return None, None
        
        print(f"Using '{target_col}' as the target column for prediction.")
        
        # Handling missing values
        dataset.fillna(dataset.median(), inplace=True)
        
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        
        # Normalize features to avoid bias towards CGPA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    except Exception as e:
        print("Error loading data:", e)
        return None, None

# Adjusting feature importance
from sklearn.feature_selection import SelectFromModel

def train_best_model(X, y):
    models = {
        "XGBoost": XGBRegressor(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "ElasticNet": ElasticNet()
    }
    
    best_model = None
    best_score = -np.inf
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"{name} - Mean R²: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    
    print(f"Best model is {best_model.__class__.__name__} with R² of {best_score:.4f}")
    
    # Hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    final_model = grid_search.best_estimator_
    return final_model

# Run the model training
X, y, scaler = load_and_prepare_data()
if X is not None and y is not None:
    model = train_best_model(X, y)
    joblib.dump(model, 'admission_prediction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully.")
