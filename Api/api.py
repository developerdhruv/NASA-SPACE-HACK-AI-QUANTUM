import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import os


from pinn.main import SeismicPINN, prepare_training_data
from gans.train import Generator, generate_samples
from vaee.train import VAE, detect_anomalies
# Load environment variables


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to checkpoint files
PinnPath = '../pinn/seismic_pinn_checkpoint.pth'
GanPath ='../gans/generator_model.pth'
VaePath ='../Vaee/vae_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# Load models
try:
    pinn_model = SeismicPINN([2, 50, 50, 50, 1])
    pinn_model.load_state_dict(torch.load(PinnPath)['model_state_dict'])
    pinn_model.eval()

    gan_generator = Generator(latent_dim=100, output_dim=100)
    gan_generator.load_state_dict(torch.load(GanPath))
    gan_generator.eval()

    vae_model = VAE(input_dim=100, hidden_dim=128, latent_dim=20)
    vae_model.load_state_dict(torch.load(VaePath))
    vae_model.eval()

    pinn_time_scaler = joblib.load('../Pinn/scalers/time_scaler.pkl')
    pinn_value_scaler = joblib.load('../Pinn/scalers/value_scaler.pkl')
    gan_scaler = joblib.load('../Gans/scaler.pkl')
    vae_scaler = joblib.load('../Vae/scaler.pkl')

    logger.info("All models and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or scalers: {str(e)}")
    raise

class SeismicData(BaseModel):
    values: List[float]
    timestamps: List[float]

@app.post("/process_seismic_data/")
async def process_seismic_data(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded CSV file
        df = pd.read_csv(file.file)
        if 'Value' not in df.columns:
            raise ValueError("CSV file must contain a 'Value' column")
        
        seismic_values = df['Value'].values
        timestamps = df.index.values

        # Step 1: PINN processing
        scaled_timestamps = pinn_time_scaler.transform(timestamps.reshape(-1, 1)).flatten()
        scaled_values = pinn_value_scaler.transform(seismic_values.reshape(-1, 1)).flatten()

        x, t, u = prepare_training_data(scaled_timestamps, scaled_values)
        xt = torch.cat([x, t], dim=1).to(device)
        
        with torch.no_grad():
            denoised_data_scaled = pinn_model(xt).cpu().numpy().flatten()
        
        denoised_data = pinn_value_scaler.inverse_transform(denoised_data_scaled.reshape(-1, 1)).flatten()

        # Step 2: GAN processing
        gan_input = gan_scaler.transform(denoised_data.reshape(-1, 1)).flatten()
        gan_sequences = []
        for i in range(0, len(gan_input) - 100 + 1, 1):
            gan_sequences.append(gan_input[i:i+100])
        gan_sequences = torch.FloatTensor(gan_sequences).to(device)
        
        num_samples = 10
        latent_dim = 100
        generated_samples = generate_samples(gan_generator, num_samples, latent_dim, device)
        augmented_data = np.concatenate([gan_sequences.cpu().numpy(), generated_samples])

        # Step 3: VAE processing
        vae_input = torch.FloatTensor(augmented_data).to(device)
        anomalies, original_data, reconstructed_data = detect_anomalies(vae_model, vae_input, device, vae_scaler)

        # Prepare results
        results = {
            "denoised_data": denoised_data.tolist(),
            "generated_samples": gan_scaler.inverse_transform(generated_samples).tolist(),
            "anomalies": anomalies.tolist(),
            "reconstructed_data": vae_scaler.inverse_transform(reconstructed_data).tolist()
        }

        logger.info("Seismic data processed successfully")
        return results
    except Exception as e:
        logger.error(f"Error processing seismic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Seismic Detection API"}

