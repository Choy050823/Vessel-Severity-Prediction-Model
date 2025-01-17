import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for input_ids, attention_mask, numerical_features, batch_y in train_loader:
            input_ids = torch.clamp(input_ids, min=0, max=30521)  # Clip to valid range
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, numerical_features, batch_y in val_loader:
                input_ids = torch.clamp(input_ids, min=0, max=30521)  # Clip to valid range
                outputs = model(input_ids, attention_mask, numerical_features)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break

        # Learning rate scheduling
        scheduler.step(val_loss)