import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # Final classification
        output = self.fc(attended_output)
        output = self.sigmoid(output)
        
        return output.squeeze()

def train_rnn_attack(sequences, labels, device, args):
    """Train RNN with attention for membership inference"""
    print("Training RNN attack model...")
    
    # Split data
    n_samples = sequences.size(0)
    indices = torch.randperm(n_samples)
    train_size = int(0.8 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_sequences = sequences[train_indices]
    train_labels = labels[train_indices]
    val_sequences = sequences[val_indices]
    val_labels = labels[val_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(train_sequences, train_labels)
    val_dataset = TensorDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = AttentionRNN(input_size=4, hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        for batch_sequences, batch_labels in train_loader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{args.dataset}/seqmia/best_rnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    
    return model 