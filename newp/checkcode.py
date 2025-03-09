import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv("expandedanhanced1.csv")



# Split the data into features and target
X = dataset.drop(columns=['Chance of Admit '])
y = dataset['Chance of Admit ']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model with optimized hyperparameters
model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R^2 Score: {r2:.2f}")

# Function to take user input and make a prediction
def predict_admission():
    gre_included = input("Do you want to provide GRE score? (yes/no): ").strip().lower()
    
    toefl_score = float(input("Enter TOEFL Score: "))
    university_rating = int(input("Enter University Rating (1-5): "))
    sop = float(input("Enter SOP strength (1.0-5.0): "))
    lor = float(input("Enter LOR strength (1.0-5.0): "))
    cgpa = float(input("Enter CGPA: "))
    research = int(input("Enter Research experience (0 for No, 1 for Yes): "))
    
    if gre_included == 'yes':
        gre_score = float(input("Enter GRE Score: "))
        input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
    else:
        input_data = np.array([[0, toefl_score, university_rating, sop, lor, cgpa, research]])  # Use 0 for GRE if not provided
    
    prediction = model.predict(input_data)[0]
    
    prediction = max(0, min(1, prediction))  # Ensure valid range
    print(f"Predicted Chance of Admission: {prediction:.2f}")

# Run the prediction function
if __name__ == "__main__":
    predict_admission()
