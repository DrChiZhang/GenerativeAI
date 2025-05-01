import torch 
import torch.nn as nn

import numpy as np
from tqdm import tqdm 

from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt

'''
Parameters:
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used for computations.")
else:
    raise RuntimeError("No GPU available. Please ensure a compatible GPU is accessible.")

data_path = './data'
batch_size = 100

x_dim = 784 # 28*28
hidden_dim = 400
latent_dim = 20
num_epochs = 50

learning_rate = 1e-3

'''
Data Loading:
'''
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(root=data_path, transform=transform, train=True, download=True)
test_dataset = MNIST(root=data_path, transform=transform, train=False, download=True)

train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
A simple VAE model: Guassian encoder and decoder
The encoder maps the input to a Gaussian distribution in latent space, and the decoder samples from this distribution to reconstruct the input.
'''
class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of the Gaussian distribution
        self.fc_var = nn.Linear(hidden_dim, latent_dim)  # Log variance of the Gaussian distribution

        self.LeakyReLU = nn.LeakyReLU(0.2)   # Leaky ReLU activation function
        self.training = True  # Set to True for training mode

    def forward(self, x):
        h_ = self.LeakyReLU(self.fc1(x))  # Apply Leaky ReLU activation function
        h_ = self.LeakyReLU(self.fc21(h_))  # Apply Leaky ReLU activation function

        mu = self.fc_mu(h_)  # Mean of the Gaussian distribution
        logvar = self.fc_var(h_)  # Log variance of the Gaussian distribution

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, x_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  # Fully connected layer from latent space to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer from hidden layer to output layer
        self.fc3 = nn.Linear(hidden_dim, x_dim)  # Fully connected layer to reconstruct the input

        self.LeakyReLU = nn.LeakyReLU(0.2)  # Leaky ReLU activation function

    def forward(self, z):
        h_ = self.LeakyReLU(self.fc1(z))  # Apply Leaky ReLU activation function
        h_ = self.LeakyReLU(self.fc2(h_))  # Apply Leaky ReLU activation function

        x_recon = torch.sigmoid(self.fc3(h_))  # Sigmoid activation function for output layer

        return x_recon
    
class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(x_dim, hidden_dim, latent_dim)  # Encoder module
        self.decoder = Decoder(latent_dim, hidden_dim, x_dim)  # Decoder module

    """
    Perform the reparameterization trick to allow for backpropagation through stochastic variables.
    Parameters:
    - mu: Mean of the latent Gaussian distribution.
    - logvar: Log variance of the latent Gaussian distribution.
    Returns:
    - A sample from the distribution N(mu, sigma^2).
    Note:
    The sampleing z ~ N(mu, sigma^2) = mu + sigma * epsilon, where epsilon ~ N(0, 1).  
    - randomness is moved into epsilon 
    - the new sampleing z = mu + sigma * epsilon is differentiable w.r.t. mu and sigma.
    - this allows us to backpropagate through the sampling process.
    The standard deviation is calculated as exp(logvar / 2) to ensure it is positive.
    """
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation from log variance
        eps = torch.randn_like(std).to(device)  # Random noise
        return mu + eps * std  # Reparameterization trick
    """
    Python knowledge:
    torch.randn_like(std) generates a tensor of the same shape as std, filled with random numbers from a normal distribution.
    torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False)-> Tensor
    Inputs:
    - input: The input tensor whose shape is to be matched.
    - dtype: The desired data type of the returned tensor. If specified, the returned tensor will have this data type.
    - layout: The desired layout of the returned tensor. If specified, the returned tensor will have this layout.
    - device: The desired device of the returned tensor. If specified, the returned tensor will be allocated on this device.
    - requires_grad: If True, the returned tensor will have gradient tracking enabled. This is useful for autograd.
    Returns:
    - A tensor of the same shape as input, filled with random numbers from a normal distribution.
    """

    def forward(self, x):
        mu, logvar = self.encoder(x)  # Encode the input to get mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from the Gaussian distribution in latent space
        x_recon = self.decoder(z)  # Decode the sampled latent vector to reconstruct the input

        return x_recon, mu, logvar
    
model = VAE(x_dim, hidden_dim, latent_dim).to(device)  # Initialize the VAE model
BEC_loss = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

def loss_function(x_recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_recon, x.view(-1, x_dim))  # Calculate the reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # Calculate the KL divergence loss
    return BCE + KLD  # Return the total loss (reconstruction loss + KL divergence loss)

'''
Training the VAE:
'''
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0  # Initialize training loss

    for batch_idx, (data, _) in enumerate(tqdm(train_dataset)):
        data = data.view(-1, x_dim).to(device)  # Flatten the input and move to device
        optimizer.zero_grad()  # Zero the gradients

        x_recon, mu, logvar = model(data)  # Forward pass through the VAE
        loss = loss_function(x_recon, data, mu, logvar)  # Calculate the loss
        loss.backward()  # Backpropagation
        train_loss += loss.item()  # Accumulate training loss
        optimizer.step()  # Update weights

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_dataset.dataset):.4f}')  # Print average loss per epoch
    torch.save(model.state_dict(), f'./ckpt/vae_epoch_{epoch+1}.pth')  # Save the model state

'''
Moel Evaluation:
'''
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loss = 0.0  # Initialize test loss
    for batch_idx, (data, _) in enumerate(tqdm(test_dataset)):
        data = data.view(-1, x_dim).to(device)  # Flatten the input and move to device

        x_recon, mu, logvar = model(data)  # Forward pass through the VAE
        loss = loss_function(x_recon, data, mu, logvar)  # Calculate the loss
        test_loss += loss.item()  # Accumulate test loss

    print(f'Test Loss: {test_loss/len(test_dataset.dataset):.4f}')  # Print average test loss per epoch

def visualize_img(x, idx):
    # Ensure the tensor is on the CPU and has the right shape for visualization
    # Assuming x is a 2D tensor where the first dimension is the number of images
    x = x.view(-1, 1, 28, 28)  # Reshape the input to image format (batch_size, channels, height, width)

    plt.figure(figsize=(4, 4))  # Size of the figure
    img = x[idx].cpu().detach().numpy().squeeze()  # Remove the channel dimension and transfer data to CPU if necessary

    plt.imshow(img)  # Display the image with a grayscale color map
    plt.title(f'Image index: {idx}')  # Add a title for clarification
    plt.axis('off')  # Turn off axis labels for a cleaner look
    plt.show()  # Render the plot

visualize_img(x_recon, idx=0)  # Visualize the reconstructed images
visualize_img(data, idx=0)  # Visualize the original images

with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(device)
    generated_images = model.decoder(noise)  # Generate new images from random noise
    generated_images = generated_images.view(-1, 1, 28, 28)  # Reshape the generated images for visualization
    save_image(generated_images, './outputs/generated_images.png', nrow=10)  # Save the generated images in a grid format

visualize_img(generated_images, idx=0)  # Visualize the generated images
visualize_img(generated_images, idx=1)  # Visualize the generated images