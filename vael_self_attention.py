import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#---------- Multi-head self-Attention Block --------
"""
Multi-head self-attention block for 2D data.
This block computes self-attention over the input feature maps, allowing the model to focus on different parts of the input simultaneously.
"""
class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        assert in_channels % heads == 0, "in_channels must be divisible by heads"
        self.heads = heads
        self.head_dim = in_channels // heads
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out   = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        def reshape_heads(tensor):
            # [B, C, H, W] -> [B, heads, head_dim, H, W]
            return tensor.view(B, self.heads, self.head_dim, H, W)
        
        q = reshape_heads(self.query_conv(x))
        k = reshape_heads(self.key_conv(x))
        v = reshape_heads(self.value_conv(x))
        
        # flatten spatial
        q = q.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        k = k.view(B, self.heads, self.head_dim, H * W)                      # [B, heads, head_dim, HW]
        v = v.view(B, self.heads, self.head_dim, H * W)                      # [B, heads, head_dim, HW]
        
        attn = torch.softmax(torch.matmul(q, k) / (self.head_dim ** 0.5), dim=-1)  # [B, heads, HW, HW]
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))   # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        out = self.gamma * out + x
        return out
    
# -------- Self-Attention Block --------
"""
Self-attention block for 2D data.
The core idea of self-attention is to compute a weighted sum of the input features, where the weights are determined by the similarity between the features.
-- Attention mechanism allows a NN to dynamically focus on different parts of the input data, based on the learned importance;
-- In image, attention models how pixels/features at different locdations interact;
-- Instead of treating all sptial locations equally, attention lets the model aggregate information from anywhere across the image. 
"""
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        """
        Q = query: “Which locations should I look at?”
        K = key: “Which locations contain relevant information?”
        V = value: “What information is present there?”
        """
        """
        Python knowledge:
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters). here, it is in_channels // 8. Both "query" and "key" projections reduce that to C//8 channels.
        kernel_size: Size of the convolving kernel. here, it is 1x1, which means it is just a per-pixel, per-channel linear transformation.
        """
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        """
         trainable scaling factor (starts at 0) used in the final output mixing.
        """
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query  = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # [B, H*W, C//8]
        proj_key    = self.key_conv(x).view(B, -1, H*W)                     # [B, C//8, H*W]
        """
        The energy matrix is computed by multiplying the query and key matrices. This gives a measure of how much focus each position should have on every other position.
        The result is a matrix of shape [B, H*W, H*W], where each element represents the attention score between two positions in the input feature map.
        """
        energy      = torch.bmm(proj_query, proj_key)                       # [B, H*W, H*W]  
        """
        The attention scores are normalized using softmax to ensure they sum to 1. This gives a probability distribution over the input positions.
        The attention matrix is of shape [B, H*W, H*W], where each row represents the attention distribution for a specific position in the input feature map.
        """
        attention   = torch.softmax(energy, dim=-1)                         # [B, H*W, H*W]     
        proj_value  = self.value_conv(x).view(B, -1, H*W)                   # [B, C, H*W]
        """
        The output is computed by multiplying the attention matrix with the value matrix. 
        This gives a weighted sum of the input features, where the weights are determined by the attention scores.
        The output is reshaped back to the original input shape.
        """
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))             # [B, C, H*W]
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

"""
Python knowledge:
torch.bmm(input, mat2, out=None) → Tensor
Batch matrix-matrix product of matrices stored in input and mat2.
input: (b, n, m) and mat2: (b, m, p) will produce (b, n, p).
The batch size is the first dimension, and the last two dimensions are treated as matrices.
"""
# -------- VAE Model --------
LATENT_DIM = 16

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1) # 28x28→14x14
        """
        Adding a sef-attention block after the first convolutional layer.
        -- Amplifying important features: the encoder can assign higher weight to sigificant or characteristic regions, such as digit strockes in MNIST;
        -- Modeling long-range dependencies: the encoder can learn to relate features from different parts of the image, allowing it to capture global context and relationships between pixels.
        -- Reducing information bottleneck: the encoder can learn to compress information more effectively by focusing on relevant features, leading to a more informative latent representation.
        """
        self.attn = MultiHeadSelfAttention2D(32, heads=4)
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