import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matlab.engine
from FPDRANet import FPDRANet

# -----------------------------------------------------------------------------
# 1. Setup and Initialization
# -----------------------------------------------------------------------------

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Start the MATLAB engine for image preprocessing
eng = matlab.engine.start_matlab()

# -----------------------------------------------------------------------------
# 2. Data Loading and Preparation
# -----------------------------------------------------------------------------

# Load data from pickle files
with open(os.path.join('data', 'FPI_data.pkl'), 'rb') as f:
    training_data = pickle.load(f)
with open(os.path.join('data', 'distribution_data.pkl'), 'rb') as f:
    distribution_data = pickle.load(f)

# Extract data and labels from loaded structures
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

# Create TensorDataset and DataLoader for testing
dataset = TensorDataset(
    noisy_images_tensor,
    label_images_tensor,
    noisy_distribution_images_tensor,
    label_distribution_images_tensor
)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# -----------------------------------------------------------------------------
# 3. Output Configuration
# -----------------------------------------------------------------------------

# Directory containing the checkpoints
# Note: Ensure this path is correct for the deployment environment
checkpoint_dir = Path(r'model')
best_model_path = checkpoint_dir / 'best_model.pth'

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = checkpoint_dir / current_time
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 4. Model Initialization and Loading
# -----------------------------------------------------------------------------

# Initialize model, loss functions, and optimizer
model = FPDRANet().to(device)
criterion = nn.MSELoss()
criterion_KL = nn.KLDivLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the best model checkpoint
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
best_val_loss = checkpoint['val_loss']

# Set the model to evaluation mode
model.eval()

# -----------------------------------------------------------------------------
# 5. Testing Loop
# -----------------------------------------------------------------------------

test_loss = 0.0

with torch.no_grad():
    for i, (noisy_img, label_img, noisy_distribution, label_distribution) in enumerate(test_loader):

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
        # Calculate min and max values
        min_val = noisy_img.min()
        max_val = noisy_img.max()
        # Normalize to range [0, 1]
        noisy_img = (noisy_img - min_val) / (max_val - min_val)

        # Reshape and prepare input tensors
        noisy_img_tensor = noisy_img.unsqueeze(1).to(device)
        combined_input = torch.cat((noisy_img_tensor, noisy_img_preprocessed_tensor), dim=1)

        # Permute dimensions for distribution data
        noisy_distribution = noisy_distribution.permute(0, 3, 2, 1)
        label_distribution = label_distribution.permute(0, 3, 2, 1)

        # -----------------------------------------------------------
        # Forward Pass and Post-processing
        # -----------------------------------------------------------
        # Make predictions (adding channel dimension implicitly handled by model/input)
        outputs, X, Y = model(noisy_img_tensor, noisy_distribution)

        # Post-process outputs (Max projection along channel dim)
        max_outputs, _ = torch.max(outputs, dim=1, keepdim=True)

        # Normalize max_outputs to range [0, 1]
        min_val = max_outputs.min()
        max_val = max_outputs.max()
        max_outputs = (max_outputs - min_val) / (max_val - min_val)

        # -----------------------------------------------------------
        # Loss Calculation
        # -----------------------------------------------------------
        label_img = label_img.unsqueeze(1)
        loss_mse = criterion(max_outputs, label_img)

        # KL Divergence requires log-probabilities
        outputs = F.log_softmax(outputs, dim=1)
        # Normalize label distribution
        label_distribution = label_distribution / label_distribution.sum(dim=1, keepdim=True)
        loss_kl = criterion_KL(outputs, label_distribution)

        # Combined Loss
        combined_loss = loss_kl * 0.001 + loss_mse
        test_loss += combined_loss.item()

        # -----------------------------------------------------------
        # Result Saving
        # -----------------------------------------------------------
        for j, output in enumerate(max_outputs):
            output_img = output.squeeze().cpu().detach().numpy()

            # Save as PNG image
            plt.imsave(
                f"{output_dir}/epoch_{epoch + 1}_batch_{i + 1}_img_{j + 1}.png",
                output_img,
                cmap='gray'
            )

            # Save as MAT file
            savemat(
                f"{output_dir}/test_batch_{i + 1}_img_{j + 1}.mat",
                {'output_img': output_img}
            )

# Calculate and print average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss}")