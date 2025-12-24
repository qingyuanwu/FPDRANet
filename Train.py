import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms.functional import gaussian_blur

import matlab.engine
from FPDRANet import FPDRANet


def ssim(img1, img2, window_size=3, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = gaussian_blur(img1, kernel_size=[window_size, window_size])
    mu2 = gaussian_blur(img2, kernel_size=[window_size, window_size])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_blur(img1 * img1, kernel_size=[window_size, window_size]) - mu1_sq
    sigma2_sq = gaussian_blur(img2 * img2, kernel_size=[window_size, window_size]) - mu2_sq
    sigma12 = gaussian_blur(img1 * img2, kernel_size=[window_size, window_size]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])


def ssim_loss(output, target):
    """
    Calculates the SSIM loss (1 - SSIM).
    """
    return 1 - ssim(output, target)


# -----------------------------------------------------------------------------
# 1. Setup and Initialization
# -----------------------------------------------------------------------------

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Start MATLAB engine for image preprocessing
eng = matlab.engine.start_matlab()

# -----------------------------------------------------------------------------
# 2. Data Loading and Preparation
# -----------------------------------------------------------------------------

# Load training data from pickle files
with open('xxxx.pkl', 'rb') as f:
    training_data = pickle.load(f)
with open('xxxx.pkl', 'rb') as f:
    distribution_data = pickle.load(f)

# Prepare lists for data and labels
noisy_images = [item[0] for item in training_data]
label_images = [item[1] for item in training_data]
noisy_distribution_images = [item[0] for item in distribution_data]
label_distribution_images = [item[1] for item in distribution_data]

# Convert lists to PyTorch tensors
noisy_images_tensor = torch.tensor(noisy_images, dtype=torch.float32)
label_images_tensor = torch.tensor(label_images, dtype=torch.float32)
noisy_distribution_images_tensor = torch.tensor(noisy_distribution_images, dtype=torch.float32)
label_distribution_images_tensor = torch.tensor(label_distribution_images, dtype=torch.float32)

# Move tensors to the configured device (GPU/CPU)
noisy_images_tensor = noisy_images_tensor.to(device)
label_images_tensor = label_images_tensor.to(device)
noisy_distribution_images_tensor = noisy_distribution_images_tensor.to(device)
label_distribution_images_tensor = label_distribution_images_tensor.to(device)

# Create TensorDataset and Split into Train/Validation Sets
dataset = TensorDataset(
    noisy_images_tensor,
    label_images_tensor,
    noisy_distribution_images_tensor,
    label_distribution_images_tensor
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------------------------------------------------------
# 3. Model Configuration
# -----------------------------------------------------------------------------

# Initialize model, loss functions, and optimizer
model = FPDRANet().to(device)
criterion = nn.MSELoss()
criterion_KL = nn.KLDivLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------------------------------------
# 4. Output Directories Configuration
# -----------------------------------------------------------------------------

# Create a subdirectory in 'outputs' using the current timestamp
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join('outputs', current_time)
os.makedirs(output_dir, exist_ok=True)

# Directory for saving model checkpoints
checkpoint_dir = os.path.join('outputs', current_time, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Directory for saving result figures
save_figure_dir = os.path.join('outputs', current_time, 'save_figure')
os.makedirs(save_figure_dir, exist_ok=True)

# Initialize tracking variables
best_val_loss = float('inf')
best_model_state = None
train_losses = []
val_losses = []

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------

num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for noisy_img, label_img, noisy_distribution, label_distribution in train_loader:
        optimizer.zero_grad()

        # -----------------------------------------------------------
        # Preprocessing using MATLAB Engine
        # -----------------------------------------------------------
        # Convert tensor to numpy array and move to CPU
        noisy_img_np = noisy_img.squeeze().cpu().numpy()
        noisy_img_preprocessed = []

        for img in noisy_img_np:
            # Convert to MATLAB double type
            img_matlab = eng.double(img)
            # Call MATLAB preprocessing function
            img_preprocessed = eng.preprocess_image(img_matlab)
            noisy_img_preprocessed.append(np.array(img_preprocessed))

        noisy_img_preprocessed = np.stack(noisy_img_preprocessed)

        # Convert back to PyTorch tensor and move to device
        noisy_img_preprocessed_tensor = torch.tensor(
            noisy_img_preprocessed, dtype=torch.float32
        ).unsqueeze(1).to(device)

        # -----------------------------------------------------------
        # Data Normalization
        # -----------------------------------------------------------
        # Calculate min and max values in the tensor
        min_val = noisy_img.min()
        max_val = noisy_img.max()
        # Normalize to range [0, 1]
        noisy_img = (noisy_img - min_val) / (max_val - min_val)

        # Reshape noisy_img to (B, 1, H, W)
        noisy_img_tensor = noisy_img.unsqueeze(1).to(device)

        # Concatenate inputs along the channel dimension (Commented out in original logic)
        combined_input = torch.cat((noisy_img_tensor, noisy_img_preprocessed_tensor), dim=1)

        # Permute dimensions for distribution data to match model input format
        noisy_distribution = noisy_distribution.permute(0, 3, 2, 1)
        label_distribution = label_distribution.permute(0, 3, 2, 1)

        # -----------------------------------------------------------
        # Forward Pass
        # -----------------------------------------------------------
        outputs, X, Y = model(noisy_img_tensor, noisy_distribution)

        # Post-process outputs (Max projection along channel dim)
        max_outputs, _ = torch.max(outputs, dim=1, keepdim=True)

        # Normalize max_outputs to range [0, 1]
        min_val = max_outputs.min()
        max_val = max_outputs.max()
        max_outputs = (max_outputs - min_val) / (max_val - min_val)

        # Prepare labels
        label_img = label_img.unsqueeze(1)

        # -----------------------------------------------------------
        # Loss Calculation
        # -----------------------------------------------------------
        loss_mse = criterion(max_outputs, label_img)

        # KL Divergence requires log-probabilities or normalized inputs
        outputs = F.log_softmax(outputs, dim=1)
        X = F.log_softmax(X, dim=1)
        Y = F.log_softmax(Y, dim=1)

        # Normalize label distribution
        label_distribution = label_distribution / label_distribution.sum(dim=1, keepdim=True)

        loss_ssim = ssim_loss(max_outputs, label_img)
        loss_kl_X = criterion_KL(X, label_distribution)
        loss_kl_Y = criterion_KL(Y, label_distribution)
        loss_kl_out = criterion_KL(outputs, label_distribution)

        # Combined Loss
        combined_loss = (
                loss_kl_out * 0.0001 +
                loss_mse +
                loss_ssim * 0.0001 +
                loss_kl_Y * 0 +
                loss_kl_X * 0
        )

        # Backward Pass and Optimization
        combined_loss.backward()
        optimizer.step()
        running_loss += combined_loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss}")

    # -----------------------------------------------------------------------------
    # 6. Validation Loop
    # -----------------------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (noisy_img, label_img, noisy_distribution, label_distribution) in enumerate(val_loader):

            # --- Preprocessing (MATLAB) ---
            noisy_img_np = noisy_img.squeeze().cpu().numpy()
            noisy_img_preprocessed = []
            for img in noisy_img_np:
                img_matlab = eng.double(img)
                img_preprocessed = eng.preprocess_image(img_matlab)
                noisy_img_preprocessed.append(np.array(img_preprocessed))

            noisy_img_preprocessed = np.stack(noisy_img_preprocessed)
            noisy_img_preprocessed_tensor = torch.tensor(
                noisy_img_preprocessed, dtype=torch.float32
            ).unsqueeze(1).to(device)

            # --- Normalization ---
            min_val = noisy_img.min()
            max_val = noisy_img.max()
            noisy_img = (noisy_img - min_val) / (max_val - min_val)
            noisy_img_tensor = noisy_img.unsqueeze(1).to(device)
            combined_input = torch.cat((noisy_img_tensor, noisy_img_preprocessed_tensor), dim=1)

            noisy_distribution = noisy_distribution.permute(0, 3, 2, 1)
            label_distribution = label_distribution.permute(0, 3, 2, 1)

            # --- Forward Pass ---
            outputs, X, Y = model(noisy_img_tensor, noisy_distribution)
            max_outputs, _ = torch.max(outputs, dim=1, keepdim=True)

            min_val = max_outputs.min()
            max_val = max_outputs.max()
            max_outputs = (max_outputs - min_val) / (max_val - min_val)

            label_img = label_img.unsqueeze(1)

            # --- Loss Calculation ---
            loss_mse = criterion(max_outputs, label_img)

            outputs = F.log_softmax(outputs, dim=1)
            X = F.log_softmax(X, dim=1)
            Y = F.log_softmax(Y, dim=1)
            label_distribution = label_distribution / label_distribution.sum(dim=1, keepdim=True)

            loss_kl_X = criterion_KL(X, label_distribution)
            loss_kl_Y = criterion_KL(Y, label_distribution)
            loss_kl_out = criterion_KL(outputs, label_distribution)
            loss_ssim = ssim_loss(max_outputs, label_img)

            combined_loss = (
                    loss_kl_out * 0.0001 +
                    loss_mse +
                    loss_ssim * 0.0001 +
                    loss_kl_Y * 0 +
                    loss_kl_X * 0
            )

            val_loss += combined_loss.item()

        # Save output images every 10 epochs
        if (epoch + 1) % 10 == 0:
            for j, output in enumerate(max_outputs):
                output_img = output.squeeze().cpu().detach().numpy()
                plt.imsave(
                    f"{save_figure_dir}/epoch_{epoch + 1}_batch_{i + 1}_img_{j + 1}.png",
                    output_img,
                    cmap='gray'
                )
                savemat(
                    f"{save_figure_dir}/epoch_{epoch + 1}_batch_{i + 1}_img_{j + 1}.mat",
                    {'output_img': output_img}
                )

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss}")

    # -----------------------------------------------------------------------------
    # 7. Checkpointing
    # -----------------------------------------------------------------------------

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth")

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(best_model_state, f"{checkpoint_dir}/best_model.pth")

    # Save loss history to .mat files
    savemat(f"{checkpoint_dir}/train_losses.mat", {'train_losses': np.array(train_losses)})
    savemat(f"{checkpoint_dir}/val_losses.mat", {'val_losses': np.array(val_losses)})

print("Training completed.")