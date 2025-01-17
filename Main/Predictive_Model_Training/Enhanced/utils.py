import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import RobertaTokenizer

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

    # Encode categorical labels (e.g., 'adjusted_severity')
    label_encoder = LabelEncoder()
    df['adjusted_severity'] = label_encoder.fit_transform(df['adjusted_severity'])

    # Encode 'detainable_deficiency' column
    if df['detainable_deficiency'].dtype == 'object':  # If it contains categorical data
        df['detainable_deficiency'] = LabelEncoder().fit_transform(df['detainable_deficiency'])

    # Separate features and labels
    text_columns = ['deficiency_finding', 'description_overview', 'immediate_causes', 'root_cause_analysis', 'VesselGroup']
    numerical_columns = ['detainable_deficiency', 'age']
    X_text = df[text_columns]  # Text features
    X_numerical = df[numerical_columns]  # Numerical features
    y = df['adjusted_severity']  # Labels

    return X_text, X_numerical, y

def create_dataloaders(X_text, X_numerical, y, batch_size=32):
    """
    Create PyTorch DataLoader objects for training and validation.
    """
    # Split data into training and validation sets
    X_text_train, X_text_val, X_numerical_train, X_numerical_val, y_train, y_val = train_test_split(
        X_text, X_numerical, y, test_size=0.2, random_state=42
    )

    # Tokenize text data using RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_text_encodings = tokenizer(X_text_train['deficiency_finding'].tolist(), truncation=True, padding=True, max_length=512)
    val_text_encodings = tokenizer(X_text_val['deficiency_finding'].tolist(), truncation=True, padding=True, max_length=512)

    # Normalize numerical data
    scaler = StandardScaler()
    X_numerical_train = scaler.fit_transform(X_numerical_train)
    X_numerical_val = scaler.transform(X_numerical_val)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(train_text_encodings['input_ids']),
        torch.tensor(train_text_encodings['attention_mask']),
        torch.tensor(X_numerical_train, dtype=torch.float32),  # Ensure numerical features are float32
        torch.tensor(y_train.values, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_text_encodings['input_ids']),
        torch.tensor(val_text_encodings['attention_mask']),
        torch.tensor(X_numerical_val, dtype=torch.float32),  # Ensure numerical features are float32
        torch.tensor(y_val.values, dtype=torch.long)
    )

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader