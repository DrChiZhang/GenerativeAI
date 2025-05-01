import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -------- Self-Attention Block --------
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query  = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        proj_key    = self.key_conv(x).view(B, -1, H*W)
        energy      = torch.bmm(proj_query, proj_key)
        attention   = torch.softmax(energy, dim=-1)
        proj_value  = self.value_conv(x).view(B, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# -------- VAE Model --------
LATENT_DIM = 16

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1) # 28x28→14x14
        self.attn = SelfAttention(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1) # 14x14→7x7
        self.fc_mu = nn.Linear(64*7*7, LATENT_DIM)
        self.fc_logvar = nn.Linear(64*7*7, LATENT_DIM)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B,32,14,14]
        x = self.attn(x)               # self-attention
        x = F.relu(self.conv2(x))      # [B,64,7,7]
        x = x.view(x.size(0), -1)      # flatten
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 64*7*7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 7x7→14x14
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 14x14→28x28

    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# -------- Loss Function --------
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# -------- Data Loader --------
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

# -------- Training Loop --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
num_epochs = 5

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for i, (data, _) in enumerate(mnist_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}] Loss: {loss.item()/len(data):.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}] Average loss: {train_loss / len(mnist_loader.dataset):.4f}')

# -------- Optional: Visualize Reconstructions --------
vae.eval()
with torch.no_grad():
    sample = next(iter(mnist_loader))[0].to(device)[:8]
    recon, _, _ = vae(sample)
    recon = recon.cpu().numpy()
    sample = sample.cpu().numpy()

fig, axes = plt.subplots(2, 8, figsize=(10, 2.5))
for i in range(8):
    axes[0,i].imshow(sample[i][0], cmap='gray')
    axes[0,i].axis('off')
    axes[1,i].imshow(recon[i][0], cmap='gray')
    axes[1,i].axis('off')
axes[0,0].set_ylabel("Input")
axes[1,0].set_ylabel("Recon")
plt.tight_layout()
plt.show()