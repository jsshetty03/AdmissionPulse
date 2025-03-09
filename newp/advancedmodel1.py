import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (using the adm_data.csv filename from the document)
dataset = pd.read_csv("/Users/jashanshetty29/newp/expandedanhanced1.csv")

# Exploratory Data Analysis
def perform_eda(df):
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Feature distribution
    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(num_features):
        if feature != 'Serial No.':  # Skip the serial number
            plt.subplot(3, 3, i)
            sns.histplot(df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    # Scatter plots of key features vs target
    plt.figure(figsize=(15, 10))
    features_to_plot = ['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research']
    for i, feature in enumerate(features_to_plot):
        plt.subplot(2, 3, i+1)
        sns.scatterplot(x=feature, y='Chance of Admit ', data=df)
        plt.title(f'{feature} vs Admission Chance')
    plt.tight_layout()
    plt.savefig('feature_vs_target.png')
    
    return corr_matrix

# Feature Engineering
def preprocess_data(df):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Drop the Serial No. column as it's just an identifier
    if 'Serial No.' in processed_df.columns:
        processed_df = processed_df.drop('Serial No.', axis=1)
    
    # Creating new features that might help the model
    processed_df['GRE_TOEFL_Ratio'] = processed_df['GRE Score'] / processed_df['TOEFL Score']
    processed_df['CGPA_Scaled'] = processed_df['CGPA'] / 10  # Scale CGPA to 0-1 range
    processed_df['GRE_CGPA_Interaction'] = processed_df['GRE Score'] * processed_df['CGPA_Scaled']
    processed_df['TOEFL_CGPA_Interaction'] = processed_df['TOEFL Score'] * processed_df['CGPA_Scaled']
    processed_df['SOP_LOR_Avg'] = (processed_df['SOP'] + processed_df['LOR ']) / 2
    
    # Feature to capture if research is present with high academics
    processed_df['Research_CGPA_Interaction'] = processed_df['Research'] * processed_df['CGPA']
    
    return processed_df

# Model Selection and Evaluation
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[model_name] = {
            'model': model,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV R2': cv_scores.mean()
        }
        
        print(f"{model_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  CV R2 Score: {cv_scores.mean():.4f}")
        print("-" * 50)
    
    return results

# Function to tune hyperparameters of the best model
def tune_model(X_train, y_train, best_model_name):
    if best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
        model = RandomForestRegressor(random_state=42)
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    else:  # Default to XGBoost if model not specified
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'gamma': [0, 0.1, 0.2]
        }
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Use a reduced parameter grid for quicker execution
    reduced_param_grid = {k: param_grid[k][:2] for k in list(param_grid.keys())[:3]}
    
    print(f"Tuning hyperparameters for {best_model_name}...")
    grid_search = GridSearchCV(model, reduced_param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Function to analyze and visualize feature importance
def analyze_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        print("\nFeature Importance:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    else:
        print("This model doesn't support feature importance analysis.")

# Function to make predictions
def predict_admission_chances(model, scaler, feature_names):
    print("\n=== Graduate Admission Chance Predictor ===\n")
    
    # Initialize input data dictionary
    input_data = {}
    
    # Get user input for each required feature
    input_data['GRE Score'] = float(input("Enter GRE Score (260-340): "))
    input_data['TOEFL Score'] = float(input("Enter TOEFL Score (90-120): "))
    input_data['University Rating'] = float(input("Enter University Rating (1-5): "))
    input_data['SOP'] = float(input("Enter Statement of Purpose Strength (1.0-5.0): "))
    input_data['LOR '] = float(input("Enter Letter of Recommendation Strength (1.0-5.0): "))
    input_data['CGPA'] = float(input("Enter CGPA (0-10): "))
    input_data['Research'] = int(input("Do you have research experience? (0 for No, 1 for Yes): "))
    
    # Create derived features
    input_data['GRE_TOEFL_Ratio'] = input_data['GRE Score'] / input_data['TOEFL Score']
    input_data['CGPA_Scaled'] = input_data['CGPA'] / 10
    input_data['GRE_CGPA_Interaction'] = input_data['GRE Score'] * input_data['CGPA_Scaled']
    input_data['TOEFL_CGPA_Interaction'] = input_data['TOEFL Score'] * input_data['CGPA_Scaled']
    input_data['SOP_LOR_Avg'] = (input_data['SOP'] + input_data['LOR ']) / 2
    input_data['Research_CGPA_Interaction'] = input_data['Research'] * input_data['CGPA']
    
    # Create DataFrame with the correct order of features
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure correct feature order
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Ensure the prediction is in the valid range [0, 1]
    prediction = max(0, min(1, prediction))
    
    print("\n" + "="*50)
    print(f"Predicted Chance of Admission: {prediction:.4f} ({prediction*100:.2f}%)")
    
    # Provide interpretation
    if prediction >= 0.85:
        print("Interpretation: Very high chance of admission.")
    elif prediction >= 0.7:
        print("Interpretation: High chance of admission.")
    elif prediction >= 0.5:
        print("Interpretation: Moderate chance of admission.")
    elif prediction >= 0.3:
        print("Interpretation: Low chance of admission.")
    else:
        print("Interpretation: Very low chance of admission.")
    
    # Suggestions for improvement
    print("\nSuggestions for improving admission chances:")
    
    # Use feature importance to provide targeted suggestions
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_names, model.feature_importances_))
        
        # CGPA suggestion
        if 'CGPA' in importances and importances['CGPA'] > 0.1 and input_data['CGPA'] < 9.0:
            print("- Try to improve your CGPA. This is a very significant factor.")
        
        # GRE suggestion
        if 'GRE Score' in importances and importances['GRE Score'] > 0.1 and input_data['GRE Score'] < 320:
            print("- Consider retaking the GRE to achieve a higher score.")
        
        # TOEFL suggestion
        if 'TOEFL Score' in importances and importances['TOEFL Score'] > 0.1 and input_data['TOEFL Score'] < 110:
            print("- Improving your TOEFL score could significantly boost your chances.")
        
        # Research suggestion
        if 'Research' in importances and importances['Research'] > 0.05 and input_data['Research'] == 0:
            print("- Gaining research experience would be beneficial.")
        
        # SOP and LOR suggestions
        if ('SOP' in importances and importances['SOP'] > 0.05 and input_data['SOP'] < 4.0) or \
           ('LOR ' in importances and importances['LOR '] > 0.05 and input_data['LOR '] < 4.0):
            print("- Work on improving your SOP and securing stronger letters of recommendation.")

# Main function
def main():
    print("="*50)
    print("Graduate Admission Prediction Model")
    print("="*50)
    
    # Load the dataset
    dataset = pd.read_csv("adm_data.csv")
    
    # Perform Exploratory Data Analysis
    print("\nPerforming Exploratory Data Analysis...")
    corr_matrix = perform_eda(dataset)
    
    # Preprocess the data
    print("\nPreprocessing the data...")
    processed_data = preprocess_data(dataset)
    
    # Split into features and target
    X = processed_data.drop(columns=['Chance of Admit '])
    y = processed_data['Chance of Admit ']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    print("\nSplitting data: 80% training, 20% testing")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Evaluate different models
    print("\nEvaluating different models...")
    model_results = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Find the best model based on R2 score
    best_model_name = max(model_results, key=lambda k: model_results[k]['R2'])
    print(f"\nBest model: {best_model_name} with R2 Score: {model_results[best_model_name]['R2']:.4f}")
    
    # Tune the hyperparameters of the best model
    best_model = tune_model(X_train, y_train, best_model_name)
    
    # Final evaluation on the test set
    y_pred = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\nFinal Model Evaluation:")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    
    # Analyze feature importance
    analyze_feature_importance(best_model, X.columns)
    
    # Save the model and scaler
    joblib.dump(best_model, 'admission_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nModel and scaler saved to disk.")
    
    # Prediction interface
    while True:
        try:
            predict_admission_chances(best_model, scaler, X.columns)
            another = input("\nDo you want to try another prediction? (yes/no): ").strip().lower()
            if another != 'yes':
                break
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("Please try again with valid inputs.")

if __name__ == "__main__":
    main()