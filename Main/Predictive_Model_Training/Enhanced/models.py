import torch
import torch.nn as nn
from transformers import RobertaModel

class TextCNN(nn.Module):
    def __init__(self, embed_dim, num_classes, num_numerical_features):
        super(TextCNN, self).__init__()
        
        # Embedding layer (for tokenized input)
        self.embedding = nn.Embedding(30522, embed_dim)  # RoBERTa vocab size is 30522
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # Fully connected layer (combines CNN output and numerical features)
        self.fc = nn.Linear(64 + num_numerical_features, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, numerical_features):
        # Debug: Check if input_ids are within the valid range
        if torch.max(input_ids) >= 30522 or torch.min(input_ids) < 0:
            raise ValueError(f"input_ids contain invalid indices. Max: {torch.max(input_ids)}, Min: {torch.min(input_ids)}")
        
        # Step 1: Generate embeddings from input_ids
        # input_ids shape: (batch_size, sequence_length)
        x = self.embedding(input_ids)  # (batch_size, sequence_length, embed_dim)
        
        # Step 2: Permute for Conv1d
        # Conv1d expects input shape: (batch_size, embed_dim, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, sequence_length)
        
        # Step 3: Apply CNN layers
        x = self.relu(self.conv1(x))  # (batch_size, 128, sequence_length)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))  # (batch_size, 64, sequence_length)
        x = self.dropout(x)
        
        # Step 4: Global average pooling
        x = x.mean(dim=2)  # (batch_size, 64)
        
        # Step 5: Concatenate numerical features
        # numerical_features shape: (batch_size, num_numerical_features)
        x = torch.cat([x, numerical_features], dim=1)  # (batch_size, 64 + num_numerical_features)
        
        # Step 6: Fully connected layer
        x = self.fc(x)  # (batch_size, num_classes)
        
        return x

class TextLSTM(nn.Module):
    def __init__(self, embed_dim, num_classes, num_numerical_features):
        super(TextLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(30522, embed_dim)  # RoBERTa vocab size is 30522
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256 + num_numerical_features, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass for the LSTM model.
        
        Args:
            input_ids: Tokenized input (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            numerical_features: Numerical features (batch_size, num_numerical_features)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Step 1: Generate embeddings from input_ids
        x = self.embedding(input_ids)  # (batch_size, sequence_length, embed_dim)
        
        # Step 2: Pass through LSTM
        x, _ = self.lstm(x)  # (batch_size, sequence_length, 256)
        
        # Step 3: Global average pooling
        x = x.mean(dim=1)  # (batch_size, 256)
        
        # Step 4: Concatenate numerical features
        x = torch.cat([x, numerical_features], dim=1)  # (batch_size, 256 + num_numerical_features)
        
        # Step 5: Apply dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)
        
        return x

class TextTransformer(nn.Module):
    def __init__(self, num_classes, num_numerical_features):
        super(TextTransformer, self).__init__()
        
        # Load pre-trained RoBERTa model
        self.transformer = RobertaModel.from_pretrained('roberta-base')
        
        # Fully connected layer for classification
        self.fc = nn.Linear(768 + num_numerical_features, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass for the transformer model.
        
        Args:
            input_ids: Tokenized input (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            numerical_features: Numerical features (batch_size, num_numerical_features)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Pass input through the transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, 768)
        
        # Apply global average pooling
        pooled_output = last_hidden_state.mean(dim=1)  # (batch_size, 768)
        
        # Concatenate numerical features
        pooled_output = torch.cat([pooled_output, numerical_features], dim=1)  # (batch_size, 768 + num_numerical_features)
        
        # Apply dropout and fully connected layer
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)  # (batch_size, num_classes)
        
        return logits