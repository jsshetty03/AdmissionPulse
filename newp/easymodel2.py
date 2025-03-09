import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv("expandedanhanced1.csv")

dataset.rename(columns=lambda x: x.strip(), inplace=True)  # Clean column names

# Split the data into features and target
X = dataset.drop(columns=['Chance of Admit'])
y = dataset['Chance of Admit']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the CatBoost model with optimized hyperparameters
model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, verbose=0, random_state=42)
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
    else:
        gre_score = 0  # Use 0 for GRE if not provided
    
    input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)[0]
    
    # Adjust prediction based on university rating
    if university_rating >= 4 and cgpa > 8.5 and toefl_score > 100:
        prediction += 0.05  # Higher chance for a strong profile in a high-rated university
    elif university_rating <= 2 and cgpa > 8.0:
        prediction += 0.10  # Easiest admission for a strong profile in a lower-rated university
    elif university_rating >= 4 and cgpa < 7.5:
        prediction -= 0.05  # Lower chances for a weaker profile in a high-rated university
    elif university_rating <= 2 and cgpa < 7.0:
        prediction -= 0.10  # Lowest chances for a weak profile in a low-rated university
    
    prediction = max(0, min(1, prediction))  # Ensure valid range
    print(f"Predicted Chance of Admission: {prediction:.2f}")

# Run the prediction function
if __name__ == "__main__":
    predict_admission()
