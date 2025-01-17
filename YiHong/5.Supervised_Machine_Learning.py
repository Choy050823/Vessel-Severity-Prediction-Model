import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv(r'Data\4.cleansed_data.csv')

# Define columns
columns_to_fill = ["description_overview", "immediate_causes"]
relevant_features = ["deficiency_finding", "root_cause_analysis"]

# Ensure relevant features exist
data[relevant_features] = data[relevant_features].fillna("")

# Function to preprocess data and train models
def fill_unknowns(df, columns_to_fill, relevant_features):
    for column in columns_to_fill:
        print(f"Processing column: {column}")

        # Split into rows with and without 'UNKNOWN'
        known_data = df[df[column] != "UNKNOWN"]
        unknown_data = df[df[column] == "UNKNOWN"]

        if known_data.empty:
            print(f"No data available to train for column: {column}")
            continue

        # Features and target
        X_known = known_data[relevant_features]
        y_known = known_data[column]

        # Combine features into a single text string
        X_known_combined = X_known.apply(lambda row: " ".join(row), axis=1)

        # Handle missing labels
        y_known = y_known.fillna("")

        # Split training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X_known_combined, y_known, test_size=0.2, random_state=42)

        # Build a pipeline for text vectorization and classification
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)
        print(f"Model accuracy for {column}: {pipeline.score(X_test, y_test):.2f}")

        # Predict for rows with 'UNKNOWN'
        if not unknown_data.empty:
            X_unknown = unknown_data[relevant_features].apply(lambda row: " ".join(row), axis=1)
            predicted = pipeline.predict(X_unknown)

            # Fill the 'UNKNOWN' values with predictions
            df.loc[df[column] == "UNKNOWN", column] = predicted

    return df

# Fill 'UNKNOWN' values in the dataset
filled_data = fill_unknowns(data, columns_to_fill, relevant_features)

# Save the updated data to a new CSV file
filled_data.to_csv(os.path.join('Data', '5.filled_data.csv'), index=False)

# Display the filtered DataFrame
print(filled_data)