import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import TextCNN, TextLSTM, TextTransformer
from train import train_model
from ensemble import majority_voting, meta_learner, evaluate_ensemble
from utils import load_and_preprocess_data, create_dataloaders

def main():
    # Load and preprocess data
    filepath = 'C:/Users/choym/OneDrive/Desktop/Vessel-Severity-Prediction/Main/Cleansed_Data/predicted_severity.csv'
    X_text, X_numerical, y = load_and_preprocess_data(filepath)
    train_loader, val_loader = create_dataloaders(X_text, X_numerical, y)

    # Define models
    num_numerical_features = X_numerical.shape[1]  # Number of numerical features
    cnn_model = TextCNN(embed_dim=256, num_classes=3, num_numerical_features=num_numerical_features)
    lstm_model = TextLSTM(embed_dim=256, num_classes=3, num_numerical_features=num_numerical_features)
    transformer_model = TextTransformer(num_classes=3, num_numerical_features=num_numerical_features)
    
 # Train models
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler_cnn = ReduceLROnPlateau(optimizer_cnn, mode='min', patience=3, factor=0.1)
    scheduler_lstm = ReduceLROnPlateau(optimizer_lstm, mode='min', patience=3, factor=0.1)
    scheduler_transformer = ReduceLROnPlateau(optimizer_transformer, mode='min', patience=3, factor=0.1)

    print("Training CNN...")
    train_model(cnn_model, train_loader, val_loader, criterion, optimizer_cnn, scheduler_cnn)

    print("Training LSTM...")
    train_model(lstm_model, train_loader, val_loader, criterion, optimizer_lstm, scheduler_lstm)

    print("Training Transformer...")
    train_model(transformer_model, train_loader, val_loader, criterion, optimizer_transformer, scheduler_transformer)

    # Generate predictions
    def generate_predictions(model, dataloader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for input_ids, attention_mask, numerical_features, _ in dataloader:
                if isinstance(model, TextLSTM):
                    outputs = model(input_ids, numerical_features)
                else:
                    outputs = model(input_ids, attention_mask, numerical_features)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        return predictions

    cnn_predictions = generate_predictions(cnn_model, val_loader)
    lstm_predictions = generate_predictions(lstm_model, val_loader)
    transformer_predictions = generate_predictions(transformer_model, val_loader)

    # Combine predictions
    ensemble_predictions = majority_voting([cnn_predictions, lstm_predictions, transformer_predictions])

    # Evaluate ensemble
    y_val = [y for _, _, _, y in val_loader.dataset]
    evaluate_ensemble(y_val, ensemble_predictions)

if __name__ == "__main__":
    main()