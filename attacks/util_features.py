import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils.loader import load_model, load_dataset

def extract_features(model, data_loader, device):
    """Extract features (loss, max_prob, std_prob, entropy) from model"""
    model.eval()
    model = model.to(device)
    features = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting features"):
            data, target = data.to(device), target.to(device)
            
            # Get model predictions
            output = model(data)
            probs = F.softmax(output, dim=1)
            
            # Calculate loss
            loss = F.cross_entropy(output, target, reduction='none')
            
            # Max probability
            max_prob = torch.max(probs, dim=1)[0]
            
            # Standard deviation of probabilities
            std_prob = torch.std(probs, dim=1)
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Combine features
            batch_features = torch.stack([loss, max_prob, std_prob, entropy], dim=1)
            features.append(batch_features.cpu())
    
    return torch.cat(features, dim=0)

def get_membership_data(args, shadow_idx=0):
    """Get members and non-members from shadow model"""
    print(f"Loading shadow model {shadow_idx} for membership data...")
    
    # Load shadow model
    shadow_model = load_model(args)
    shadow_path = f"{args.dataset}/{shadow_idx}/shadow_model.pt"
    shadow_checkpoint = torch.load(shadow_path, map_location='cpu')
    shadow_model.load_state_dict(shadow_checkpoint)
    
    # Load membership masks
    shadow_masks_path = f"{args.dataset}/{shadow_idx}/shadow_keep.npy"
    shadow_masks = np.load(shadow_masks_path)
    
    # Load shadow data
    shadow_data = load_dataset(args, data_type="shadow", augment=False)
    
    # Split into members and non-members
    member_indices = np.where(shadow_masks == 1)[0]
    nonmember_indices = np.where(shadow_masks == 0)[0]
    
    member_data = torch.utils.data.Subset(shadow_data, member_indices)
    nonmember_data = torch.utils.data.Subset(shadow_data, nonmember_indices)
    
    return member_data, nonmember_data, shadow_model

def create_sequence_features(seqmia_dir, member_data, nonmember_data, device, args):
    """Create sequence features for training the RNN"""
    print("Creating sequence features...")
    
    # Load all snapshots
    snapshot_paths = sorted([f for f in os.listdir(seqmia_dir) if f.startswith('snapshot_') and f.endswith('.pt')])
    models = []
    
    for snapshot_path in snapshot_paths:
        model = load_model(args).to(device)
        checkpoint = torch.load(os.path.join(seqmia_dir, snapshot_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)
    
    # Create data loaders
    member_loader = DataLoader(member_data, batch_size=128, shuffle=False, num_workers=4)
    nonmember_loader = DataLoader(nonmember_data, batch_size=128, shuffle=False, num_workers=4)
    
    # Extract features for members
    member_features = []
    for model in tqdm(models, desc="Extracting member features"):
        features = extract_features(model, member_loader, device)
        member_features.append(features)
    
    # Extract features for non-members
    nonmember_features = []
    for model in tqdm(models, desc="Extracting non-member features"):
        features = extract_features(model, nonmember_loader, device)
        nonmember_features.append(features)
    
    # Stack features to create sequences
    member_sequences = torch.stack(member_features, dim=1)  # (n_samples, seq_len, 4)
    nonmember_sequences = torch.stack(nonmember_features, dim=1)  # (n_samples, seq_len, 4)
    
    # Create labels
    member_labels = torch.ones(member_sequences.size(0))
    nonmember_labels = torch.zeros(nonmember_sequences.size(0))
    
    # Combine data
    all_sequences = torch.cat([member_sequences, nonmember_sequences], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)
    
    return all_sequences, all_labels 