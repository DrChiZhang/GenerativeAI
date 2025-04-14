import  torch 
import torch.nn as nn
import torch.optim as optim

x = torch.rand(100, 10) # 100 samples, 10 features
y = torch.randint(0, 2, (100,)) # 100 samples, binary labels

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=2):
        """Initialize the model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output classes.
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()               # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer
        self.softmax = nn.Softmax(dim=1)    # Softmax for output layer  

    def forward(self, x):
        x = self.fc1(x)               # Input layer to hidden layer
        x = self.relu(x)               # Activation function
        x = self.fc2(x)               # Hidden layer to output layer    
        x = self.softmax(x)            # Softmax for output layer
        return x
    
# Initialize the model, loss function and optimizer
model = SimpleMLP(input_size=10, hidden_size=50, output_size=2)
# Use a cross-entropy loss function for binary classification and an Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x)  # Pass the input through the model
    loss = loss_fn(outputs, y)  # Calculate the loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_data = torch.rand(5, 10)  # 5 samples for testing
    predictions = model(test_data)  # Get predictions
    predicted_classes = torch.argmax(predictions, dim=1)  # Get class with highest probability
    print(f'Test Data Predictions: {predicted_classes}')  # Print predicted classes