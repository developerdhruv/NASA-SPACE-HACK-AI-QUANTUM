import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pinn.dataload import load_seismic_data, prepare_training_data

checkpoint_path = 'seismic_pinn_checkpoint.pth'

# Set the device to GPU if available
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeismicPINN(nn.Module):
    def __init__(self, layers):
        super(SeismicPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

def denoising_loss(model, xt, u_noisy):
    u_pred = model(xt)
    return torch.mean((u_pred - u_noisy)**2)

def pde_loss(model, x, t, c=1.0):
    # Ensure both x and t require gradients
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    pde_residual = u_t - c * u_xx
    return torch.mean(pde_residual**2)

def train_pinn(xt, u_clean, u_noisy, num_epochs=10000, learning_rate=0.001):
    model = SeismicPINN([2, 50, 50, 50, 1]).to(Device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    xt, u_clean, u_noisy = xt.to(Device), u_clean.to(Device), u_noisy.to(Device)
    x, t = xt[:, 0].reshape(-1, 1).to(Device), xt[:, 1].reshape(-1, 1).to(Device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        loss_pde = pde_loss(model, x, t, c=1.0)
        loss_denoise = denoising_loss(model, xt, u_noisy)
        
        total_loss = loss_pde + 5 * loss_denoise
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item()}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    return model

def visualize_results(model, xt, u_clean, u_noisy, value_scaler, metadata):
    with torch.no_grad():
        u_pred = model(xt.to(Device)).cpu()
    
    # Inverse transform the data back to original scale
    u_clean = value_scaler.inverse_transform(u_clean.cpu().numpy())
    u_noisy = value_scaler.inverse_transform(u_noisy.cpu().numpy())
    u_pred = value_scaler.inverse_transform(u_pred.numpy())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original Signal - {metadata['Station']} {metadata['Channel']}")
    plt.plot(xt[:, 1].cpu(), u_clean)
    plt.xlabel('Normalized Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 3, 2)
    plt.title("Noisy Signal")
    plt.plot(xt[:, 1].cpu(), u_noisy)
    plt.xlabel('Normalized Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 3, 3)
    plt.title("Denoised Signal (PINN)")
    plt.plot(xt[:, 1].cpu(), u_pred)
    plt.xlabel('Normalized Time')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    file_path = '../seiss_data.csv'  # Update this path
    x, t, u, value_scaler, metadata = load_seismic_data(file_path)
    xt, u_clean, u_noisy = prepare_training_data(x, t, u)

    # Convert data to GPU (if available)
    xt, u_clean, u_noisy = xt.to(Device), u_clean.to(Device), u_noisy.to(Device)

    # Train the model
    model = train_pinn(xt, u_clean, u_noisy)

    # Visualize results
    visualize_results(model, xt, u_clean, u_noisy, value_scaler, metadata)