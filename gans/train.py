import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# GAN models
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def load_and_preprocess_data(file_path, sequence_length=100, stride=1):
    df = pd.read_csv(file_path)
    seismic_values = df['Value'].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    seismic_values_scaled = scaler.fit_transform(seismic_values.reshape(-1, 1)).flatten()

    sequences = []
    for i in range(0, len(seismic_values_scaled) - sequence_length + 1, stride):
        sequences.append(seismic_values_scaled[i:i+sequence_length])

    return torch.FloatTensor(sequences), scaler

def train_gan(generator, discriminator, dataloader, num_epochs=200, latent_dim=100, device='cpu'):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_seismic in enumerate(dataloader):
            real_seismic = real_seismic[0]
            batch_size = real_seismic.size(0)
            real_seismic = real_seismic.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)

            output_real = discriminator(real_seismic)
            d_loss_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_seismic = generator(noise)
            output_fake = discriminator(fake_seismic.detach())
            d_loss_fake = criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            
            output = discriminator(fake_seismic)
            g_loss = criterion(output, label_real)
            
            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return generator, discriminator

def generate_samples(generator, num_samples, latent_dim, device):
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        generated_samples = generator(noise).cpu().numpy()
    return generated_samples

def plot_real_vs_generated(real_samples, generated_samples, scaler):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(scaler.inverse_transform(real_samples[0].reshape(-1, 1)))
    plt.title("Real Seismic Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(scaler.inverse_transform(generated_samples[0].reshape(-1, 1)))
    plt.title("Generated Seismic Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    file_path = '../seiss_data.csv'  # Update this path
    sequences, scaler = load_and_preprocess_data(file_path)
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize GAN
    latent_dim = 100
    sequence_length = sequences.shape[1]
    generator = Generator(latent_dim, sequence_length).to(device)
    discriminator = Discriminator(sequence_length).to(device)

    # Train GAN
    generator, discriminator = train_gan(generator, discriminator, dataloader, num_epochs=200, latent_dim=latent_dim, device=device)

    # Generate samples
    num_samples = 10
    generated_samples = generate_samples(generator, num_samples, latent_dim, device)

    # Plot real vs generated samples
    plot_real_vs_generated(sequences.numpy(), generated_samples, scaler)

    # Save the trained models and scaler
    torch.save(generator.state_dict(), 'generator_model.pth')
    torch.save(discriminator.state_dict(), 'discriminator_model.pth')
    joblib.dump(scaler, 'gan_scaler.pkl')
    print("Models and scaler saved successfully.")

def load_models_and_scaler():
    generator = Generator(latent_dim=100, output_dim=100)  # Adjust dimensions as needed
    discriminator = Discriminator(input_dim=100)  # Adjust dimension as needed
    
    generator.load_state_dict(torch.load('generator_model.pth'))
    discriminator.load_state_dict(torch.load('discriminator_model.pth'))
    scaler = joblib.load('gan_scaler.pkl')
    
    return generator, discriminator, scaler

if __name__ == "__main__":
    main()

    # Example of loading models and scaler
    # loaded_generator, loaded_discriminator, loaded_scaler = load_models_and_scaler()
    # print("Models and scaler loaded successfully.")