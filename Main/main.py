import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import importlib.util
import sys
import os

def load_and_preprocess_data(filepath):
    """
    Load the dataset and perform basic preprocessing.
    """
    # Load the dataset
    df = pd.read_csv(filepath)

    # Filter relevant columns
    relevant_columns = [
        'deficiency_finding', 
        'description_overview', 
        'immediate_causes', 
        'root_cause_analysis',
        'VesselGroup',
        'detainable_deficiency',
        'age',
        'adjusted_severity'
    ]
    df = df[relevant_columns]

    # Drop rows with missing values
    df = df.dropna(subset=relevant_columns)

    return df

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

def get_bert_embeddings(df):
    torch.cuda.empty_cache()
    """
    Generate BERT embeddings for text columns.
    """
    # Combine text columns into a single column
    text_columns = ['deficiency_finding', 'description_overview', 'immediate_causes', 'root_cause_analysis', 'VesselGroup']
    df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

    # Function to get BERT embeddings in smaller batches
    def _get_embeddings(texts, batch_size=16, max_length=512):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

    # Get BERT embeddings for the combined text
    text_embeddings = _get_embeddings(df['combined_text'].tolist(), batch_size=16, max_length=512)

    return text_embeddings

def main():
    # Step 0: Import Libraries from other folders
    # Add the Predictive_Model_Training folder to sys.path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training'))

    # Dynamically import modules
    def import_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # Import 1_DataPreprocessing
    data_preprocessing = import_module(
        "data_preprocessing",
        os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training', '1_DataPreprocessing.py')
    )
    load_and_preprocess_data = data_preprocessing.load_and_preprocess_data

    # Import 2_FeatureEngineering
    feature_engineering = import_module(
        "feature_engineering",
        os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training', '2_FeatureEngineering.py')
    )
    engineer_features = feature_engineering.engineer_features

    # Import 3_Text_Embedding
    text_embedding = import_module(
        "text_embedding",
        os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training', '3_TextEmbedding.py')
    )
    get_bert_embeddings = text_embedding.get_bert_embeddings

    # Import 4_TrainModel
    train_model = import_module(
        "train_model",
        os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training', '4_TrainModel.py')
    )
    train_final_model = train_model.train_final_model
    # train_xgboost = train_model.train_xgboost
    
    hyperparameters = import_module(
        "hyperparameters",
        os.path.join(os.path.dirname(__file__), 'Predictive_Model_Training', '7_Hyperparameters.py')
    )
    tune_hyperparameters = hyperparameters.tune_hyperparameters

    # Step 1: Load and preprocess data
    df = load_and_preprocess_data('./Main/Cleansed_Data/final_data.csv')
    print("Load and preprocessing completed!")

    # Step 2: Perform feature engineering for non-text columns
    df, scaler = engineer_features(df)
    print("Feature engineering completed!")
    # print(df.head())

    # # Step 3: Generate BERT embeddings for text columns
    text_embeddings = get_bert_embeddings(df)
    print("Text embedding completed!")
    # print(text_embedding)

    # Step 4: Combine all features
    X = np.hstack([text_embeddings, df[['detainable_deficiency', 'age']].values])
    y = df['predicted_severity_encoded'].values

    # Step 5: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data preparation completed!")
    
    
    # Step 6: Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=20)

    # Step 7: Train the final model
    final_model = train_final_model(X_train, y_train, X_test, y_test, best_params)
    print("Final model done")

if __name__ == "__main__":
    main()