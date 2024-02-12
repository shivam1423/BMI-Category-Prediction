from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained model and scaler here
# Example: Load a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# Example: Load a MinMaxScaler
scaler = MinMaxScaler() 

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for processing the form data
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    # features  = request.json
    # features_list = [features[key] for key in sorted(features.keys())]
    # features = ['Male', 27, 'Software Engineer', 6.1, 6, 42, 6, '126/83', 77, 4200, 'None']
    columns=['Gender', 'Age', 'Occupation', 'Sleep Duration',
       'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
       'Sleep Disorder']
    features = [request.form[f'{i.lower()}'] for i in columns]

    data = pd.DataFrame([features], columns=columns)

    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True)
    # Convert the new columns to integers
    data['Systolic_BP'] = pd.to_numeric(data['Systolic_BP'])
    data['Diastolic_BP'] = pd.to_numeric(data['Diastolic_BP'])

    # Drop the original 'Blood Pressure' column
    data.drop('Blood Pressure', axis=1, inplace=True)

    # Encode categorical variables using one-hot encoding
    data = pd.get_dummies(data, columns=['Gender', 'Occupation','Sleep Disorder'])

    with open('min_max_scaler.pkl', 'rb') as file:
        loaded_min_max_scaler = pickle.load(file)

    missing_columns = set(loaded_min_max_scaler.get_feature_names_out()) - set(data.columns)
    for column in missing_columns:
        data[column] = 0 
    # Reorder columns based on the feature names of the loaded MinMaxScaler
    ordered_columns = loaded_min_max_scaler.get_feature_names_out()
    data = data[ordered_columns]
    # Scale the input features
    scaled_features = loaded_min_max_scaler.transform(data)

    with open('rf_classifier.pkl', 'rb') as f:
        rf = pickle.load(f)

    # Make a prediction using the model
    preds = rf.predict(scaled_features)
    feature_importances = rf.feature_importances_
    feature_importances = pd.Series(rf.feature_importances_, index=data.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    # Pass the prediction to the HTML template
    return render_template('index.html', prediction=preds[0],feature_importances=feature_importances.index[0])

if __name__ == '__main__':
    app.run(debug=True)
