# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## DESIGN STEPS


### STEP 1: 
Import necessary libraries including torch, torchvision, matplotlib, numpy, and torchsummary for visualization and model summary.

### STEP 2: 
Configure computing device using CUDA if available; otherwise, use CPU for model training operations.

### STEP 3: 
Apply transformations to normalize and convert MNIST dataset images into tensor format suitable for neural networks.

### STEP 4: 
Load MNIST training and testing datasets using DataLoader with specified batch size and shuffling enabled.

### STEP 5: 
Define a function to add Gaussian noise to input images and clip pixel values appropriately.

### STEP 6: 
Create the DenoisingAutoencoder class using convolutional and transpose convolutional layers for encoding and decoding.

### STEP 7: 
Implement the forward method that passes input through encoder and decoder to reconstruct denoised outputs.

### STEP 8: 
Initialize model, define Mean Squared Error (MSE) as reconstruction loss, and choose Adam optimizer for training.

### STEP 9: 
Display model architecture summary including layer dimensions using the summary function for clear structural understanding.

### STEP 10: 
Train model over multiple epochs, adding noise to inputs and minimizing reconstruction loss through backpropagation updates.

### STEP 11: 
Evaluate trained autoencoder on test dataset by generating denoised images from noisy inputs without gradient updates.

### STEP 12: 
Visualize original, noisy, and reconstructed images using Matplotlib to compare model performance qualitatively.





## PROGRAM

### Name: SARWESHVARAN A

### Register Number: 212223230198

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
print("Name : SARWESHVARAN A\nReg No : 212223230198")
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: SARWESHVARAN A")
    print("Register Number: 212223230198")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: SARWESHVARAN A")
    print("Register Number: 212223230198")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


```

### OUTPUT:

### Model Summary:

<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/ad22135f-d66e-4a19-be42-02790186e6f7" />



### Training loss:


<img width="300" height="300" alt="Screenshot 2025-11-28 190541" src="https://github.com/user-attachments/assets/62808a22-f89c-4082-90c8-3ec27dd8e6ef" />

## Original vs Noisy Vs Reconstructed Image:


<img width="1514" height="528" alt="image" src="https://github.com/user-attachments/assets/c1ab441a-84ac-461f-8a78-dd76b5415569" />



## RESULT
Thus, a convolutional autoencoder for image denoising application has been developed.
