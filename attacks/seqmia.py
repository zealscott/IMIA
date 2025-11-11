import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from utils.loader import load_dataset, load_model, load_config
from tqdm import tqdm
from attacks.util_distillation import distill_target_model
from attacks.util_features import get_membership_data, create_sequence_features, extract_features
from attacks.util_rnn import train_rnn_attack, AttentionRNN

def attack(dataset, n_shadows, test_losses):
    """Main SeqMIA attack function"""
    # Set up device and random seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    args = type('Args', (), {})()
    args.dataset = dataset
    args.n_shadows = n_shadows
    load_config(args)
    
    # Step 1: Load reference dataset for distillation
    print("Loading reference dataset...")
    reference_dataset = load_dataset(args, data_type="reference", augment=False)
    
    # Step 2: Distill target model (model 256) and create snapshots
    target_model_path = f"{dataset}/{n_shadows-1}/shadow_model.pt"
    seqmia_dir = distill_target_model(args, target_model_path, reference_dataset, device)
    
    # Step 3: Get membership data from shadow model 0
    member_data, nonmember_data, shadow_model = get_membership_data(args, shadow_idx=0)
    
    # Step 4: Create sequence features
    sequences, labels = create_sequence_features(seqmia_dir, member_data, nonmember_data, device, args)
    
    # Step 5: Train RNN attack model
    rnn_model = train_rnn_attack(sequences, labels, device, args)
    
    # Step 6: Generate test scores
    print("Generating test scores...")
    
    # Load test data
    test_dataset = load_dataset(args, data_type="target", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Load all snapshots for testing
    snapshot_paths = sorted([f for f in os.listdir(seqmia_dir) if f.startswith('snapshot_') and f.endswith('.pt')])
    test_models = []
    
    for snapshot_path in snapshot_paths:
        model = load_model(args).to(device)
        checkpoint = torch.load(os.path.join(seqmia_dir, snapshot_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_models.append(model)
    
    # Extract test features
    test_features = []
    for model in tqdm(test_models, desc="Extracting test features"):
        features = extract_features(model, test_loader, device)
        test_features.append(features)
    
    test_sequences = torch.stack(test_features, dim=1).to(device)
    
    # Generate membership scores
    rnn_model.eval()
    with torch.no_grad():
        membership_scores = rnn_model(test_sequences).cpu().numpy()
    
    print(f"Generated membership scores for {len(membership_scores)} test samples")
    print(f"Score range: [{membership_scores.min():.4f}, {membership_scores.max():.4f}]")
    
    return membership_scores
    
