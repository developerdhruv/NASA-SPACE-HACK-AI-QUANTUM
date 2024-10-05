import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

def add_noise(data, noise_level=0.05):
    return data + noise_level * torch.randn_like(data)

def pde_loss(model, x, t, c):
    xt = torch.cat([x, t], dim=1)
    u = model(xt)
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    pde_residual = u_tt - c**2 * u_xx
    return torch.mean(pde_residual**2)

def boundary_loss(model, x, t):
    xt_boundary = torch.cat([x, t], dim=1)
    u_boundary = model(xt_boundary)
    return torch.mean(u_boundary**2)

def initial_condition_loss(model, x, t, u_true):
    xt_initial = torch.cat([x, t], dim=1)
    u_initial = model(xt_initial)
    return torch.mean((u_initial - u_true)**2)

def denoising_loss(model, x, t, u_noisy, noise_level=0.05):
    xt_clean = torch.cat([x, t], dim=1)
    xt_noisy = torch.cat([add_noise(x, noise_level), add_noise(t, noise_level)], dim=1)
    
    u_clean = model(xt_clean)
    u_pred_noisy = model(xt_noisy)
    
    return torch.mean((u_pred_noisy - u_noisy)**2) + torch.mean((u_clean - u_noisy)**2)

def generate_data(num_points=1000, noise_level=0.05):
    x = torch.linspace(0, 1, num_points).reshape(-1, 1)
    t = torch.linspace(0, 1, num_points).reshape(-1, 1)
    xt = torch.cat([x, t], dim=1)
    
    # True solution (example: damped sine wave)
    u_true = torch.sin(2 * np.pi * x) * torch.exp(-t)
    
    # Add noise
    u_noisy = add_noise(u_true, noise_level)
    
    return x, t, u_true, u_noisy

def train_pinn(num_epochs=10000, learning_rate=0.001):
    model = SeismicPINN([2, 50, 50, 50, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    x, t, u_true, u_noisy = generate_data()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        loss_pde = pde_loss(model, x, t, c=1.0)
        loss_bc = boundary_loss(model, x, torch.zeros_like(x)) + boundary_loss(model, x, torch.ones_like(x))
        loss_ic = initial_condition_loss(model, x, torch.zeros_like(x), u_true[0])
        loss_denoise = denoising_loss(model, x, t, u_noisy)
        
        total_loss = loss_pde + 10 * loss_bc + 10 * loss_ic + 5 * loss_denoise
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item()}")
    
    return model

def visualize_results(model, x, t, u_true, u_noisy):
    with torch.no_grad():
        xt = torch.cat([x, t], dim=1)
        u_pred = model(xt)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("True Signal")
    plt.imshow(u_true.reshape(100, 100).T, aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Noisy Signal")
    plt.imshow(u_noisy.reshape(100, 100).T, aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Denoised Signal (PINN)")
    plt.imshow(u_pred.reshape(100, 100).T, aspect='auto', cmap='viridis')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Train the model
model = train_pinn()

# Generate data for visualization
x, t, u_true, u_noisy = generate_data()

# Visualize results
visualize_results(model, x, t, u_true, u_noisy)