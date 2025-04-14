import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. Generate synthetic data with pure PyTorch
def generate_synthetic_data(num_samples=1000, num_features=20):
    # Create random features and weights
    X = torch.randn(num_samples, num_features)
    weights = torch.randn(num_features, 1)
    
    # Generate labels using logistic regression model with noise
    logits = X @ weights + 0.1 * torch.randn(num_samples, 1)
    probabilities = torch.sigmoid(logits)
    y = (probabilities > 0.5).float().squeeze()
    
    return X, y

# Generate and split data
X, y = generate_synthetic_data()
dataset = TensorDataset(X, y)

# Split into train/test (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. Define neural network
class Classifier(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

# 3. Initialize training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop with Adam
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # Calculate epoch metrics
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print statistics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Test Acc: {100*correct/total:.2f}%")
    print("--------------------------")