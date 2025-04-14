import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Dense Layer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)

# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)

# Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)

# DenseNet for MNIST
class DenseNetMNIST(nn.Module):
    def __init__(self, num_blocks, num_layers_per_block, growth_rate, num_classes):
        super(DenseNetMNIST, self).__init__()
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(num_layers_per_block[i], num_channels, growth_rate))
            num_channels += num_layers_per_block[i] * growth_rate
            if i != num_blocks - 1:
                self.transition_layers.append(TransitionLayer(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.bn = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        for i in range(len(self.dense_blocks)):
            out = self.dense_blocks[i](out)
            if i != len(self.dense_blocks) - 1:
                out = self.transition_layers[i](out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)


# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNetMNIST(num_blocks=3, num_layers_per_block=[4, 4, 4], growth_rate=12, num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Evaluation function
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# Training and evaluation loop
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, criterion, epoch)
    evaluate(model, device, test_loader, criterion)