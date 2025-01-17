import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=20):
    """
    Tune hyperparameters using Optuna.
    """
    # Convert labels to 0-indexed format
    y_train = y_train - 1  # Subtract 1 to make labels 0-indexed
    y_test = y_test - 1    # Subtract 1 to make labels 0-indexed

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Determine the number of unique classes in the dataset
    num_classes = len(np.unique(y_train))

    def objective(trial):
        """
        Objective function for Optuna to optimize.
        """
        # Define hyperparameters to tune
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Learning rate
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Dropout rate
        hidden_dim = trial.suggest_int('hidden_dim', 64, 256)  # Hidden layer dimension

        # Define the model with tuned hyperparameters
        class HybridModel(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(HybridModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        # Initialize the model, loss function, and optimizer
        model = HybridModel(X_train.shape[1], num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(20):  # Number of epochs
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, y_pred = torch.max(outputs, dim=1)
                accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())

            # Report intermediate results to Optuna
            trial.report(accuracy, epoch)

            # Handle pruning (stop trial if it's not promising)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return accuracy

    # Run Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    # Best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    return best_params