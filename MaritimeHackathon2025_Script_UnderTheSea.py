import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the training data
train_df = pd.read_csv(r'./Main/Cleansed_Data/final_data.csv')

# Handle missing values in the training data
train_df = train_df.fillna(method='ffill')

# Convert categorical columns to numeric using Label Encoding
categorical_columns = train_df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    train_df[col] = label_encoders[col].fit_transform(train_df[col])

# Define features (X) and target (y)
X = train_df.drop(columns=['adjusted_severity'])
y = train_df['adjusted_severity']

# Normalize the numerical features (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(512, input_dim=X_train.shape[1], activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Adjusted severity is a continuous variable, so use linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save('severity_predictor_model.keras')

# Load the test data
test_df0 = pd.read_csv(r'./Main/Cleansed_Data/psc_severity_test.csv')
test_df = test_df0.copy()

# Split 'def_text' of test data into columns
test_df['deficiency_finding'] = test_df['def_text'].str.extract(r'Deficiency/Finding: (.*?)\n', expand=False)
test_df['description_overview'] = test_df['def_text'].str.extract(r'Description Overview: (.*?)\n', expand=False)
test_df['immediate_causes'] = test_df['def_text'].str.extract(r'Immediate Causes: (.*?)\n', expand=False)
test_df['root_cause_analysis'] = test_df['def_text'].str.extract(r'Root Cause Analysis: (.*?)\n', expand=False)
test_df['corrective_action'] = test_df['def_text'].str.extract(r'Corrective Action: (.*?)\n', expand=False)
test_df['preventive_action'] = test_df['def_text'].str.extract(r'Preventive Action: (.*?)\n', expand=False)
test_df['deficiency_code'] = test_df['def_text'].str.extract(r'Deficiency Code: (\d+)', expand=False)
test_df['detainable_deficiency'] = test_df['def_text'].str.extract(r'Detainable Deficiency: (\w+)', expand=False)

# Drop irrelevant columns of test data
test_df = test_df.drop('def_text', axis=1)
test_df = test_df.drop('PscAuthorityId', axis=1)
test_df = test_df.drop('VesselId', axis=1)
test_df = test_df.drop('InspectionDate', axis=1)
test_df = test_df.drop('PscInspectionId', axis=1)

# Align preprocessing for test data
for col in categorical_columns:
    if col in test_df.columns:
        test_df[col] = test_df[col].apply(lambda x: label_encoders[col].transform([x])[0]
                                          if x in label_encoders[col].classes_ else -1)

# Drop irrelevant columns and handle missing values
test_df = test_df.fillna(method='ffill')
X_test = test_df.drop(columns=['predicted_severity'], errors='ignore')  # Ensure no target column
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test data
predictions = model.predict(X_test_scaled)

# Ensure the predictions match the length of test_df0 by padding or truncating
if len(predictions) < len(test_df0):
    predictions = np.pad(predictions, (0, len(test_df0) - len(predictions)), constant_values=np.nan)
elif len(predictions) > len(test_df0):
    predictions = predictions[:len(test_df0)]

# Add the predictions as a new column in the test dataframe
test_df0['predicted_severity'] = predictions

# Save the result with predictions to a new CSV file
# test_df0.to_csv(os.path.join('Data', '11.predicted_data.csv'), index=False)

# Display the filtered DataFrame
print('Dataframe with predicted "adjusted_severity" by machine learning added')
print(test_df0)


# Define a function to categorize the values
def categorize_value(value):
    if 0 <= value <= 1.666:
        return 'Low'
    elif 1.666 < value <= 2.333:
        return 'Medium'
    elif 2.333 < value <= 4:
        return 'High'
    else:
        return value  # If out of range, keep the original value (optional)

# Specify the column name you want to process
column_name = 'predicted_severity'

# Apply the categorization function to the specific column and replace the values
test_df0[column_name] = test_df0[column_name].apply(categorize_value)

test_df0 = test_df0[['PscInspectionId', 'deficiency_code', 'predicted_severity']]
test_df0['deficiency_code'] = test_df0['deficiency_code'].astype(str).str.zfill(5)

# Save the result to a new CSV file
test_df0.to_csv('MaritimeHackathon2025_SeverityPredictions_UnderTheSea.csv', index=False)

# Print the updated dataframe
print(test_df0)