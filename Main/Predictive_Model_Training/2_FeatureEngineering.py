from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Perform feature engineering for non-text columns.
    """
    # Encode 'detainable_deficiency' column
    df['detainable_deficiency'] = df['detainable_deficiency'].map({'Yes': 1, 'No': 0})

    # Scale the 'age' column
    scaler = StandardScaler()
    df['age'] = scaler.fit_transform(df[['age']])

    # Encode the target column with custom mapping
    # severity_mapping = {
    #     'High': 3,
    #     'Medium': 2,
    #     'Low': 1
    # }
    df['predicted_severity_encoded'] = df['adjusted_severity']

    return df, scaler