import numpy as np
from utils.shadow_utils import get_all_shadow_models, get_test
import torch
import os
from utils.loader import load_config
from attacks.util_rapid_mlp import train_rapid_mlp, RAPIDMLP


def attack(dataset, n_shadows, test_losses):
    """Main RAPID attack function"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    args = type("Args", (), {})()
    args.dataset = dataset
    args.n_shadows = n_shadows
    load_config(args)

    # Step 1: Get features from all shadow models using get_all_shadow_models
    print("Getting features from all shadow models...")

    # Get all losses from shadow models
    all_losses, all_masks, n_samples = get_all_shadow_models(
        dataset, n_shadows, attack_type="rapid", data_type="shadow", model_type="shadow"
    )

    # Get test losses from target model
    test_losses, test_masks = get_test(dataset, n_shadows, attack_type="rapid", data_type="shadow", model_type="target")

    # Step 2: Create features for each instance from each shadow model
    print("Creating features for each instance from each shadow model...")
    all_features = []
    all_labels = []

    # First, compute reference losses (when instance is NOT in training set) for each instance
    print("Computing reference losses...")
    reference_losses = []
    for instance_idx in range(n_samples):
        # Get losses for this instance across all shadow models where it's NOT a member
        instance_losses = all_losses[:, instance_idx]  # Shape: (n_shadows-1,)
        instance_masks = all_masks[:, instance_idx]  # Shape: (n_shadows-1,)

        # Get losses where instance is NOT a member (mask == 0)
        non_member_losses = instance_losses[instance_masks == 0]

        reference_loss = np.mean(non_member_losses)

        reference_losses.append(reference_loss)

    reference_losses = np.array(reference_losses)

    # Now create training features
    for shadow_idx in range(all_losses.shape[0]):  # For each shadow model
        for instance_idx in range(n_samples):  # For each instance
            # Get loss for this instance from this shadow model
            shadow_loss = all_losses[shadow_idx, instance_idx]
            reference_loss = reference_losses[instance_idx]

            # Compute features
            loss_feature = shadow_loss
            calibrated_loss_feature = reference_loss - shadow_loss

            # Create 2D feature vector
            features = np.array([loss_feature, calibrated_loss_feature])
            all_features.append(np.mean(features, axis=1))

            # Get label from shadow model membership mask
            label = all_masks[shadow_idx, instance_idx]
            all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # sample 10000 features
    all_features = all_features[:10000]
    all_labels = all_labels[:10000]

    print(f"Total features: {all_features.shape}, Total labels: {all_labels.shape}")

    # Step 3: Train RAPID MLP
    rapid_model = train_rapid_mlp(all_features, all_labels, device, args)

    # Step 4: Generate test scores
    print("Generating test scores...")

    # Compute test features using the same reference losses
    test_features = []
    for instance_idx in range(len(test_losses)):
        # Compute calibrated loss
        calibrated_loss = test_losses[instance_idx] - reference_losses[instance_idx]

        # Create 2D feature vector
        features = np.array([test_losses[instance_idx], calibrated_loss])
        test_features.append(np.mean(features, axis=1))

    test_features = np.array(test_features)

    # Generate membership scores
    rapid_model.eval()
    with torch.no_grad():
        test_features_tensor = torch.FloatTensor(test_features).to(device)
        membership_scores = rapid_model(test_features_tensor).cpu().numpy()

    return membership_scores
