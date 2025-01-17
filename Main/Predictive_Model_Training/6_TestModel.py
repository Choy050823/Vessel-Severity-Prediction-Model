import joblib
import pandas as pd
import numpy as np

def test_model(new_data_path, ensemble_model, le, scaler):
    # Load new data
    new_data = pd.read_csv(new_data_path)

    # Filter relevant columns
    new_data = new_data[['Description Overview', 'Immediate Causes', 'Root Cause Analysis', 'age']]

    # Combine textual columns
    new_data['combined_text'] = new_data[['Description Overview', 'Immediate Causes', 'Root Cause Analysis']].apply(lambda x: ' '.join(x), axis=1)

    # Scale the 'age' column
    new_data['age'] = scaler.transform(new_data[['age']])

    # Extract BERT embeddings
    new_text_embeddings = get_bert_embeddings(new_data['combined_text'])

    # Combine features
    new_data_combined = np.hstack([new_text_embeddings, new_data[['age']].values])

    # Make predictions
    predictions = ensemble_model.predict(new_data_combined)

    # Decode predictions back to original labels
    predicted_labels = le.inverse_transform(predictions)

    # Add predictions to the new data
    new_data['predicted_severity'] = predicted_labels

    # Save the results
    new_data.to_csv('new_data_with_predictions.csv', index=False)

    print("Predictions saved to 'new_data_with_predictions.csv'")