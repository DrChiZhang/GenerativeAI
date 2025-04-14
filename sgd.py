import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Generate synthetic data
def make_moons(n_samples=1000, noise=0.2):
    X = torch.rand((n_samples, 2)) * 2 - 1  # Random points in [-1, 1]
    y = ((X[:, 0]**2 + X[:, 1]**2 + noise * torch.randn(n_samples)) > 0.5).long()
    return X, y

X, y = make_moons(n_samples=20, noise=0.2)

# Manually split the dataset
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the MLP
model = MLP()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 500
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

# Evaluate the model on the test dataset
test_loader = DataLoader(test_dataset, batch_size=test_size)
model.eval()
with torch.no_grad():
    X_test, y_test = next(iter(test_loader))
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

# Optional: Visualize decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01),
                            torch.arange(y_min, y_max, 0.01))
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        Z = model(grid)
        _, Z = torch.max(Z, 1)
        Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', lw=0.8)
    plt.show()

plot_decision_boundary(model, X, y)