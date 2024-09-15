import torch
import torch.nn as nn
import torch.optim as optim

class EdgeAIModel(nn.Module):
    def __init__(self):
        super(EdgeAIModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def save_pytorch_model(model, filepath='seismic_edge_model.pth'):
    """
    Save PyTorch model to a file.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Initialize the model
    model = EdgeAIModel()
    
    # Print model summary
    print(model)
    
    # Create some random data for demonstration (e.g., 100 samples with 128 features)
    test_input = torch.randn(100, 128)
    
    # Forward pass through the model
    output = model(test_input)
    print("Model output shape:", output.shape)
    
    # Save model for edge deployment
    save_pytorch_model(model, 'seismic_edge_model.pth')
