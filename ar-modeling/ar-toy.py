import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Model Definition
# -----------------
class SimpleAutoregressiveNet(nn.Module):
    """
    A toy 1D autoregressive fully-connected network using masking to enforce 
    that prediction at each timestep only depends on the current and previous inputs.
    """
    def __init__(self, seq_len, hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        
        # Linear layer to hidden
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        
        # Linear layer to output; outputs full sequence at once
        self.fc2 = nn.Linear(hidden_dim, seq_len)

        # Mask for autoregressive property: lower triangle of ones
        # mask[i, j] == 1 means input position j is allowed to influence output i
        mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len, seq_len)
        self.register_buffer('mask', mask)  # So mask moves with model.to(device)

    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, seq_len)
        Returns:
          out: (batch_size, seq_len) predictions for all time steps
        """
        h = F.relu(self.fc1(x))    # (batch_size, hidden_dim)
        out = self.fc2(h)          # (batch_size, seq_len)
        
        # Apply the causal mask so out[:, t] only depends on x up to and including t
        # Each output dimension mixes only current and previous inputs
        # The division normalizes output since some outputs will sum more masked values
        out = torch.matmul(out, self.mask) / (torch.sum(self.mask, dim=1) + 1e-8)
        return out

# -----------------
# Training Loop
# -----------------
if __name__ == "__main__":
    torch.manual_seed(0)  # For reproducibility
    
    # Create a toy dataset: batch of sequences [0, 1, 2, ..., seq_len-1], with random noise
    batch_size = 16
    seq_len = 10
    data = torch.arange(seq_len).float().repeat(batch_size, 1)  # (batch_size, seq_len)
    # Add a little Gaussian noise to each sequence element
    inputs = data + 0.1 * torch.randn_like(data)  # (batch_size, seq_len)

    # Initialize the autoregressive model
    model = SimpleAutoregressiveNet(seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training: Predict the value at t+1 given all values up to t
    for step in range(100):
        y_pred = model(inputs)  # (batch_size, seq_len)
        
        # For loss, we use y_pred[:, :-1] (predictions for timesteps 0 to n-2)
        # and inputs[:, 1:] (targets: "next value" for timesteps 1 to n-1)
        loss = F.mse_loss(y_pred[:, :-1], inputs[:, 1:])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    # -----------------
    # Testing the Model
    # -----------------
    # Give the model a noiseless test sequence and see how well it predicts
    test_seq = torch.arange(seq_len).float().unsqueeze(0)  # (1, seq_len), e.g., [0, 1, 2, ..., 9]
    model.eval()
    with torch.no_grad():
        test_pred = model(test_seq)    # (1, seq_len)
        print("Input sequence:     ", test_seq.squeeze().tolist())
        print("Predicted sequence: ", test_pred.squeeze().tolist())