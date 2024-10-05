import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=0.1):
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

def train_vae(vae, optimizer, data_loader, device, epochs=300):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        avg_loss = total_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}, Average loss: {avg_loss:.4f}')
        if avg_loss < 0.1:  # Early stopping condition
            print("Converged early. Stopping training.")
            break

def detect_anomalies(vae, data_loader, device, scaler):
    vae.eval()
    anomalies = []
    original_data = []
    reconstructed_data = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                logging.info(f"Batch {batch_idx}: Type of batch: {type(batch)}, Length: {len(batch)}")
                
                # Handle different types of batch data
                if isinstance(batch, torch.Tensor):
                    data = batch
                elif isinstance(batch, (tuple, list)):
                    if len(batch) == 0:
                        logging.warning(f"Batch {batch_idx}: Empty batch, skipping")
                        continue
                    data = batch[0]  # Assume the first element is the data
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")
                
                data = data.to(device)
                logging.info(f"Batch {batch_idx}: Input data shape: {data.shape}")
                
                if data.dim() == 1:
                    data = data.unsqueeze(0)  # Add batch dimension if missing
                elif data.dim() > 2:
                    data = data.view(data.size(0), -1)  # Flatten if more than 2D
                
                logging.info(f"Batch {batch_idx}: Reshaped data shape: {data.shape}")
                
                if data.shape[1] != vae.input_dim:
                    raise ValueError(f"Input dimension mismatch. Expected {vae.input_dim}, got {data.shape[1]}")
                
                recon, mu, logvar = vae(data)
                logging.info(f"Batch {batch_idx}: Reconstruction shape: {recon.shape}")
                
                mse_loss = nn.MSELoss(reduction='none')(recon, data)
                mse_loss = mse_loss.mean(axis=1)
                
                anomalies.extend(mse_loss.cpu().numpy())
                original_data.extend(data.cpu().numpy())
                reconstructed_data.extend(recon.cpu().numpy())
            
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                logging.error(f"Batch content: {batch}")
                raise
    
    if not anomalies:
        raise ValueError("No data was successfully processed")
    
    anomalies = np.array(anomalies)
    threshold = np.mean(anomalies) + 2 * np.std(anomalies)
    anomalies = anomalies > threshold
    
    return anomalies, np.array(original_data), np.array(reconstructed_data)

def load_and_preprocess_data(file_path, sequence_length=50, stride=1):
    df = pd.read_csv(file_path)
    seismic_values = df['Value'].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    seismic_values_scaled = scaler.fit_transform(seismic_values.reshape(-1, 1)).flatten()

    sequences = []
    for i in range(0, len(seismic_values_scaled) - sequence_length + 1, stride):
        sequences.append(seismic_values_scaled[i:i+sequence_length])

    return torch.FloatTensor(sequences), scaler

def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    file_path = '../seiss_data.csv'  # Update this path
    sequences, scaler = load_and_preprocess_data(file_path)
    dataset = TensorDataset(sequences)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize VAE
    input_dim = sequences.shape[1]
    hidden_dim = 128
    latent_dim = 20
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    # Train VAE
    train_vae(vae, optimizer, data_loader, device, epochs=300)

    # Detect anomalies
    
    try:
        anomalies, original_data, reconstructed_data = detect_anomalies(vae, data_loader, device, scaler)
    except Exception as e:
        logging.error(f"Error in anomaly detection: {str(e)}")

    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.plot(scaler.inverse_transform(original_data.reshape(-1, 1)), alpha=0.5, label='Original')
    plt.scatter(np.where(anomalies)[0], 
                scaler.inverse_transform(original_data.reshape(-1, 1))[np.where(anomalies)], 
                color='red', label='Anomalies')
    plt.title('Original Seismic Data with Detected Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot reconstruction
    plt.subplot(2, 1, 2)
    plt.plot(scaler.inverse_transform(reconstructed_data.reshape(-1, 1)), label='Reconstructed')
    plt.title('Reconstructed Seismic Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Save the model and scaler
    torch.save(vae.state_dict(), 'vae_model.pth')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    main()