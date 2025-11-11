import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils.loader import load_model, load_dataset

def compute_calibrated_loss(logits, labels, temperature=1.0):
    """Compute calibrated loss using temperature scaling"""
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute softmax probabilities
    probs = torch.softmax(scaled_logits, dim=1)
    
    # Get predicted probabilities for correct class
    batch_size = logits.size(0)
    correct_probs = probs[torch.arange(batch_size), labels]
    
    # Compute calibrated loss (negative log likelihood)
    calibrated_loss = -torch.log(correct_probs + 1e-8)
    
    return calibrated_loss

def extract_rapid_features(model, data_loader, device, temperature=1.0):
    """Extract loss and calibrated loss features from model"""
    model.eval()
    model = model.to(device)
    features = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting RAPID features"):
            data, target = data.to(device), target.to(device)
            
            # Get model predictions
            logits = model(data)
            
            # Compute regular loss
            regular_loss = nn.functional.cross_entropy(logits, target, reduction='none')
            
            # Compute calibrated loss
            calibrated_loss = compute_calibrated_loss(logits, target, temperature)
            
            # Combine features
            batch_features = torch.stack([regular_loss, calibrated_loss], dim=1)
            features.append(batch_features.cpu())
    
    return torch.cat(features, dim=0)

def get_shadow_membership_data(args, shadow_idx):
    """Get members and non-members from a shadow model"""
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