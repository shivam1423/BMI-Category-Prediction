import pandas as pd
from sklearn import preprocessing
import pickle

features = ['Male', 27, 'Software Engineer', 6.1, 6, 42, 6, '126/83', 77, 4200, 'None']
data = pd.DataFrame([features], columns=['Gender', 'Age', 'Occupation', 'Sleep Duration',
    'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
    'Sleep Disorder'])

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