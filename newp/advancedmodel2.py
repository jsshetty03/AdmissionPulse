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
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None, None
            
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns")
        print(f"Columns found: {', '.join(dataset.columns)}")
        
        # Try to identify the target column (admission chance column)
        # Look for column names containing variations of "admit" or "chance"
        target_col = None
        for col in dataset.columns:
            if 'admit' in col.lower() or 'chance' in col.lower():
                target_col = col
                break
                
        # If no target column found, ask user to specify
        if target_col is None:
            print("\nCould not automatically identify the admission chance column.")
            print("Available columns:")
            for i, col in enumerate(dataset.columns):
                print(f"{i+1}. {col}")
            col_idx = int(input("\nPlease enter the number of the column containing admission chances: ")) - 1
            target_col = dataset.columns[col_idx]
        
        print(f"Using '{target_col}' as the target column for prediction.")
        
        # Check for missing values
        missing_values = dataset.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Simple imputation for demonstration
            dataset = dataset.fillna(dataset.mean())
        
        # Split the data
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        
        # Display basic statistics
        print("\nFeature statistics:")
        print(X.describe().transpose())
        
        return X, y, target_col
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# Model selection and training
def train_best_model(X, y):
    if X is None or y is None:
        return None
        
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing and model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor())
    ])
    
    # Define models to try
    models = {
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42, max_iter=2000)
    }
    
    # Compare models with cross-validation
    print("\nComparing models with 5-fold cross-validation:")
    best_model = None
    best_score = -np.inf
    cv_results = {}
    
    for name, model in models.items():
        pipeline.set_params(model=model)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2').mean()
        cv_results[name] = cv_score
        print(f"{name} - Mean R²: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_model = name
    
    print(f"\nBest model is {best_model} with R² of {best_score:.4f}")
    
    # Hyperparameter tuning for the best model
    print(f"\nPerforming hyperparameter tuning for {best_model}...")
    
    if best_model == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5]
        }
    elif best_model == 'RandomForest':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5]
        }
    elif best_model == 'GradientBoosting':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5]
        }
    else:  # ElasticNet
        param_grid = {
            'model__alpha': [0.1, 0.5],
            'model__l1_ratio': [0.2, 0.5]
        }
    
    pipeline.set_params(model=models[best_model])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nTest set performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    try:
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Admission Chances')
        plt.tight_layout()
        plt.savefig('admission_prediction_performance.png')
        print("Performance plot saved as 'admission_prediction_performance.png'")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Feature importance
    try:
        if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
            importances = best_pipeline.named_steps['model'].feature_importances_
            feature_names = X.columns
            
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Feature importance plot saved as 'feature_importance.png'")
            
            print("\nFeature Importance:")
            for i in indices:
                print(f"{feature_names[i]}: {importances[i]:.4f}")
    except Exception as e:
        print(f"Could not generate feature importance: {e}")
    
    # Save the model
    try:
        joblib.dump(best_pipeline, 'admission_prediction_model.pkl')
        print("\nModel saved as 'admission_prediction_model.pkl'")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    return best_pipeline, X.columns

# Function to take user input and make a prediction
def predict_admission(model, feature_names=None):
    if model is None:
        print("No model available for prediction.")
        return
        
    print("\n=== Graduate Admission Prediction Tool ===")
    
    try:
        input_data = {}
        
        # If feature names not provided, try to get from model
        if feature_names is None:
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                # Default feature names based on typical dataset structure
                feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
                print("Warning: Using default feature names which may not match your model.")
                print(f"Default features: {', '.join(feature_names)}")
        
        # Print feature names for reference
        print("\nRequired input features:")
        for i, feature in enumerate(feature_names):
            print(f"{i+1}. {feature}")
        
        # Collect inputs for each feature
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if 'gre' in feature_lower:
                gre_included = input(f"\nDo you want to provide {feature}? (yes/no): ").strip().lower()
                if gre_included == 'yes':
                    value = float(input(f"Enter {feature} (260-340): "))
                    # Validate GRE score range
                    if value < 260 or value > 340:
                        print(f"Warning: {feature} typically ranges from 260 to 340.")
                    input_data[feature] = value
                else:
                    # Use median value if GRE not provided
                    input_data[feature] = 310  # Approximate median GRE score
                    print(f"Using default value of 310 for {feature}")
            
            elif 'toefl' in feature_lower:
                value = float(input(f"\nEnter {feature} (0-120): "))
                # Validate TOEFL score range
                if value < 0 or value > 120:
                    print(f"Warning: {feature} typically ranges from 0 to 120.")
                input_data[feature] = value
            
            elif 'rating' in feature_lower or 'rank' in feature_lower:
                value = int(input(f"\nEnter {feature} (1-5): "))
                # Validate rating range
                if value < 1 or value > 5:
                    print(f"Warning: {feature} should be between 1 and 5.")
                input_data[feature] = value
            
            elif 'sop' in feature_lower or 'statement' in feature_lower:
                value = float(input(f"\nEnter {feature} strength (1.0-5.0): "))
                # Validate SOP range
                if value < 1.0 or value > 5.0:
                    print(f"Warning: {feature} typically ranges from 1.0 to 5.0.")
                input_data[feature] = value
            
            elif 'lor' in feature_lower or 'recommendation' in feature_lower:
                value = float(input(f"\nEnter {feature} strength (1.0-5.0): "))
                # Validate LOR range
                if value < 1.0 or value > 5.0:
                    print(f"Warning: {feature} typically ranges from 1.0 to 5.0.")
                input_data[feature] = value
            
            elif 'cgpa' in feature_lower or 'gpa' in feature_lower:
                value = float(input(f"\nEnter {feature} (0.0-10.0): "))
                # Validate CGPA range
                if value < 0.0 or value > 10.0:
                    print(f"Warning: {feature} typically ranges from 0.0 to 10.0.")
                input_data[feature] = value
            
            elif 'research' in feature_lower:
                value = int(input(f"\nEnter {feature} experience (0 for No, 1 for Yes): "))
                # Validate research input
                if value not in [0, 1]:
                    print(f"Warning: {feature} should be 0 or 1.")
                input_data[feature] = value
            
            else:
                # Generic input for any other features
                value = float(input(f"\nEnter {feature}: "))
                input_data[feature] = value
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns match expected order
        if isinstance(feature_names, np.ndarray):
            input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Ensure prediction is within valid range
        prediction = max(0, min(1, prediction))
        
        print("\n=== Prediction Results ===")
        print(f"Predicted Chance of Admission: {prediction:.4f} ({prediction*100:.2f}%)")
        
        # Provide interpretation
        if prediction >= 0.8:
            print("Interpretation: Very high chance of admission")
        elif prediction >= 0.6:
            print("Interpretation: Good chance of admission")
        elif prediction >= 0.4:
            print("Interpretation: Moderate chance of admission")
        else:
            print("Interpretation: Lower chance of admission")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please try again with valid inputs.")

# Main execution
if __name__ == "__main__":
    print("Graduate Admission Prediction System")
    
    # Ask for dataset file path
    file_path = input("Enter the path to your dataset CSV file (or press Enter for default 'expandedanhanced1.csv'): ").strip()
    if not file_path:
        file_path = "expandedanhanced1.csv"
    
    # Option to load existing model or train new one
    use_existing = input("Use existing model if available? (yes/no): ").strip().lower()
    
    model = None
    feature_names = None
    
    if use_existing == 'yes':
        try:
            model = joblib.load('admission_prediction_model.pkl')
            print("Existing model loaded successfully.")
            
            # Try to get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
        except Exception as e:
            print(f"Could not load existing model: {e}")
            print("Training a new model...")
            X, y, target_col = load_and_prepare_data(file_path)
            if X is not None and y is not None:
                model, feature_names = train_best_model(X, y)
    else:
        X, y, target_col = load_and_prepare_data(file_path)
        if X is not None and y is not None:
            model, feature_names = train_best_model(X, y)
    
    if model is not None:
        while True:
            predict_admission(model, feature_names)
            another = input("\nMake another prediction? (yes/no): ").strip().lower()
            if another != 'yes':
                break
    
    print("Thank you for using the Graduate Admission Prediction System!")