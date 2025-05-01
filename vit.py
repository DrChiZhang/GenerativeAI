import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=32,           # Image dimension (assumes square images)
        patch_size=4,          # Size of one side of a patch
        in_channels=3,         # Usually 3 (RGB images)
        n_classes=10,          # Number of classification labels
        emb_size=64,           # Embedding size of each token/patch
        depth=4,               # Number of Transformer encoder layers
        n_heads=4,             # Number of heads in multi-head attention
        ff_dim=256,            # Feedforward network hidden size
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch + embedding
        self.patch_embed = nn.Conv2d(
            in_channels, emb_size, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token & Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP head for classification
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

        # Optional initialization (improves convergence)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                  # (B, emb_size, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)         # (B, n_patches, emb_size)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1 + n_patches, emb_size)
        x = x + self.pos_embed                         # Add positional embedding
        x = self.dropout(x)

        x = self.transformer(x)                        # (B, 1 + n_patches, emb_size)
        cls_out = x[:, 0]                              # (B, emb_size)
        out = self.head(cls_out)                       # (B, n_classes)
        return out
    
# ---- 1. Data preparation ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# ---- 2. Instantiate the model, loss, and optimizer ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(img_size=32, patch_size=4, emb_size=64, n_classes=10, depth=4, n_heads=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ---- 3. Training loop ----
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    loss_per_epoch = running_loss / total
    accuracy = correct / total
    return loss_per_epoch, accuracy

# ---- 4. Evaluation ----
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    loss_per_epoch = running_loss / total
    accuracy = correct / total
    return loss_per_epoch, accuracy

# ---- 5. Training & Validation ----
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, testloader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs} '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {test_loss:.4f} Acc: {test_acc:.4f}')