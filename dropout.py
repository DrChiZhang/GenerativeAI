import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split    
from torchvision import transforms, datasets 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define the transforations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#load data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_laoder = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the MLP model with dropout
class MLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
# Initialize the MLP with dropout
model = MLP(dropout_rate=0.5).to(device)   
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_laoder:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

# Evaluate the model on the test dataset
model.eval()
correct = 0 
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
# The above code implements a simple MLP with dropout for the MNIST dataset.