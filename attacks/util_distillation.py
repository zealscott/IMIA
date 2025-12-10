import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from utils.loader import load_model

def distill_target_model(args, target_model_path, reference_dataset, device):
    """Distill the target model and save snapshots every 10 epochs"""
    
    # Create seqmia directory
    seqmia_dir = f"{args.dataset}/seqmia"
    os.makedirs(seqmia_dir, exist_ok=True)
    
    # Check if the last snapshot (epoch 100) already exists
    last_snapshot_path = os.path.join(seqmia_dir, "snapshot_100.pt")
    if os.path.exists(last_snapshot_path):
        print("Last snapshot (epoch 100) already exists. Skipping distillation.")
        return seqmia_dir
    
    print("Loading target model for distillation...")
    
    # Load target model (model 256)
    target_model = load_model(args).to(device)
    target_checkpoint = torch.load(target_model_path, map_location=device)
    target_model.load_state_dict(target_checkpoint)
    target_model.eval()
    
    # Create distillation model
    distill_model = load_model(args).to(device)
    optimizer = torch.optim.SGD(distill_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    # Training loop
    train_loader = DataLoader(reference_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    for epoch in range(100):
        distill_model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/100")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Student predictions
            student_output = distill_model(data)
            
            # Teacher predictions
            with torch.no_grad():
                teacher_output = target_model(data)
            
            # Combined loss: CE + KL divergence
            ce = ce_loss(student_output, target)
            kl = kl_loss(F.log_softmax(student_output / 4.0, dim=1),
                        F.softmax(teacher_output / 4.0, dim=1)) * (4.0 ** 2)
            
            loss = 0.5 * ce + 0.5 * kl
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Save snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            snapshot_path = os.path.join(seqmia_dir, f"snapshot_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': distill_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, snapshot_path)
            print(f"Saved snapshot at epoch {epoch+1}")
    
    return seqmia_dir 