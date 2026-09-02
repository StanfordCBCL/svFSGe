import torch
import torch.nn as nn
import time
import os
import csv
import numpy as np

class ShearStressNN(nn.Module):
    def __init__(self, branch_dims=[25, 64, 64, 64], trunk_dims=[3, 64, 64, 64], final_dim=64, out_dim=3):
        """
        Neural network for predicting shear stress using spherical harmonic coefficients

        Args:
            branch_dims: Dimensions for the branch network (first dim should match coefficient length, default 25)
            trunk_dims: Dimensions for the trunk network (first dim should be 3 for xyz coordinates)
            final_dim: Dimension of the final combined layer
        """
        super(ShearStressNN, self).__init__()
        
        # Branch network for spherical harmonic coefficients
        modules = []
        in_channels = branch_dims[0]
        for h_dim in branch_dims[1:]:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh()
            ))
            in_channels = h_dim
        self._branch = nn.Sequential(*modules)
        
        # Trunk network for xyz coordinates
        modules = []
        in_channels = trunk_dims[0]
        for h_dim in trunk_dims[1:]:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh()
            ))
            in_channels = h_dim
        self._trunk = nn.Sequential(*modules)
        
        self.branch_to_final = nn.Linear(branch_dims[-1], final_dim)
        self.trunk_to_final = nn.Linear(trunk_dims[-1], final_dim)
        self._out_layer = nn.Linear(final_dim, out_dim)
    
    def forward(self, coeffs, xyz):
        """
        Forward pass of the network
        
        Args:
            coeffs: Spherical harmonic coefficients [batch_size, num_coeffs]
            xyz: Spatial coordinates [batch_size, 3]
            
        Returns:
            Predicted shear stress vector [batch_size, 3]
        """
        y_br = self._branch(coeffs)
        y_br = self.branch_to_final(y_br)
        
        y_tr = self._trunk(xyz)
        y_tr = self.trunk_to_final(y_tr)
        
        # Element-wise multiplication of branch and trunk outputs
        y_combined = y_br * y_tr
        
        return self._out_layer(y_combined)
    
    def loss(self, coeffs, xyz, values, reg_lambda=1):
        """
        Compute L2 loss between predictions and target values
        
        Args:
            coeffs: Spherical harmonic coefficients [batch_size, num_coeffs]
            xyz: Spatial coordinates [batch_size, 3]
            values: Target shear stress vectors [batch_size, 3]
            reg_lambda: Regularization parameter
            
        Returns:
            L2 loss value
        """
        y_pred = self.forward(coeffs, xyz)
        l2_loss = torch.mean(torch.sqrt(torch.sum((y_pred - values) ** 2, dim=1) + 1e-8), dim=0)
        return l2_loss


def train_model(model, train_loader, val_loader, n_epochs=100, lr=1e-3, results_dir='results'):
    """
    Train the shear stress neural network
    
    Args:
        model: ShearStressNN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of training epochs
        lr: Learning rate
        results_dir: Directory to save results
        
    Returns:
        Lists of training and validation losses
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up CSV file for immediate logging
    csv_path = os.path.join(results_dir, 'training_losses.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'time_elapsed_min', 'epoch_time_sec'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # GradScaler only helps on CUDA; on CPU it causes gradient overflow
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    train_losses = []
    val_losses = []
    epoch_train = []
    epoch_val = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"Training on {device} for {n_epochs} epochs")
    print(f"Results will be saved to {csv_path}")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()
        
        # Process each batch without progress bar
        for coeffs_batch, xyz_batch, values_batch in train_loader:
            coeffs_batch = coeffs_batch.to(device, non_blocking=True)
            xyz_batch = xyz_batch.to(device, non_blocking=True)
            values_batch = values_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.module.loss(coeffs_batch, xyz_batch, values_batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item() * len(coeffs_batch)
            
            # Adding training loss to the list
            train_losses.append(loss.item())
            epoch_train.append(epoch)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for coeffs_batch, xyz_batch, values_batch in val_loader:
                coeffs_batch = coeffs_batch.to(device, non_blocking=True)
                xyz_batch = xyz_batch.to(device, non_blocking=True)
                values_batch = values_batch.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = model.module.loss(coeffs_batch, xyz_batch, values_batch)
                
                total_val_loss += loss.item() * len(coeffs_batch)
                val_losses.append(loss.item())
                epoch_val.append(epoch)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        # Calculate timing information
        epoch_time = time.time() - epoch_start_time
        total_time_elapsed = (time.time() - start_time) / 60
        
        # Log to CSV immediately after each epoch
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, total_time_elapsed, epoch_time])
            
        # Save loss lists to disk
        np.save(os.path.join(results_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(results_dir, 'val_losses.npy'), np.array(val_losses))
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{n_epochs} - Time: {total_time_elapsed:.2f}m (epoch: {epoch_time:.2f}s) - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(results_dir, 'shear_stress_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'scaler_state_dict': scaler.state_dict(),
            }, model_path)
            print(f"  → New best model saved at {model_path} (val_loss: {best_val_loss:.6f})")
    
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to {csv_path}")
    print(f"Best model saved to {os.path.join(results_dir, 'shear_stress_model.pt')}")
    
    return train_losses, val_losses, epoch_val, epoch_train