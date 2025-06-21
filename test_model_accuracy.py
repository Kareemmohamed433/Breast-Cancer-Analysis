import pandas as pd
import requests
import json
import re
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("data.csv")

# Select 5 Malignant and 5 Benign samples
malignant = data[data['diagnosis'] == 'M'].sample(5, random_state=42)
benign = data[data['diagnosis'] == 'B'].sample(5, random_state=42)
test_samples = pd.concat([malignant, benign])

# Features to send to the model (30 features, excluding id and diagnosis)
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# API endpoint
url = "http://localhost:8000/predict_json"  # Change to "/predict" if JSON endpoint is unavailable

# Store predictions and actual labels
predictions = []
actual_labels = []

# Function to parse HTML response (if using /predict endpoint)
def parse_html_response(html_text):
    # Assuming prediction is in a format like "<h2>Prediction: Malignant</h2>"
    match = re.search(r'Prediction: (Malignant|Benign)', html_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

# Test each sample
for idx, row in test_samples.iterrows():
    # Prepare feature dictionary
    features = {f"feature_{i}": row[feature_columns[i]] for i in range(30)}
    
    # Send request to the model
    try:
        response = requests.post(url, data=features)
        response.raise_for_status()
        
        # Handle JSON response
        try:
            result = response.json()
            prediction = result['prediction']
        except json.JSONDecodeError:
            # Handle HTML response (if /predict endpoint)
            prediction = parse_html_response(response.text)
            if prediction is None:
                print(f"Error parsing prediction for sample {row['id']}")
                continue
        
        # Store prediction and actual label
        predictions.append(prediction)
        actual_labels.append('Malignant' if row['diagnosis'] == 'M' else 'Benign')
        
        print(f"Sample ID: {row['id']}, Actual: {row['diagnosis']}, Predicted: {prediction}")
        
    except requests.RequestException as e:
        print(f"Error for sample {row['id']}: {e}")
        continue

# Calculate accuracy
if predictions:
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
else:
    print("No valid predictions were made.")