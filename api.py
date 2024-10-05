import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import custom modules
from pinn.main import SeismicPINN
from pinn.dataload import prepare_training_data
from gans.train import Generator, generate_samples
from vaee.train import VAE, detect_anomalies

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to checkpoint files
PinnPath = './pinn/seismic_pinn_checkpoint.pth'
GanPath = './gans/generator_model.pth'
VaePath = './Vaee/vae_model.pth'

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# Load models
try:
    pinn_model = SeismicPINN([2, 50, 50, 50, 1]).to(device)
    pinn_model.load_state_dict(torch.load(PinnPath)['model_state_dict'])
    pinn_model.eval()

    gan_generator = Generator(input_dim=100, output_dim=100).to(device)
    gan_generator.load_state_dict(torch.load(GanPath))
    gan_generator.eval()

    vae_model = VAE(input_dim=50, hidden_dim=128, latent_dim=20).to(device)
    vae_model.load_state_dict(torch.load(VaePath))
    vae_model.eval()

    pinn_time_scaler = joblib.load('./pinn/scalers/time_scaler.pkl')
    pinn_value_scaler = joblib.load('./pinn/scalers/value_scaler.pkl')
    gan_scaler = joblib.load('./gans/gan_scaler.pkl')
    vae_scaler = joblib.load('./vaee/scaler.pkl')

    logger.info("All models and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or scalers: {str(e)}")
    raise

# Define Pydantic model for seismic data input
class SeismicData(BaseModel):
    values: List[float]
    timestamps: List[float]

@app.post("/process_seismic_data/")
async def process_seismic_data(file: UploadFile = File(...)):
    try:
        # Ensure that the file is in CSV format
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")

        # Read and preprocess the uploaded CSV file
        df = pd.read_csv(file.file)
        if 'Value' not in df.columns:
            raise ValueError("CSV file must contain a 'Value' column.")
        if df['Value'].isnull().any():
            raise ValueError("CSV file contains missing values in 'Value' column.")

        seismic_values = df['Value'].values
        timestamps = np.arange(len(seismic_values))  # Using index as timestamps for simplicity

        # Step 1: PINN processing
        try:
            scaled_timestamps = pinn_time_scaler.transform(timestamps.reshape(-1, 1)).flatten()
            scaled_values = pinn_value_scaler.transform(seismic_values.reshape(-1, 1)).flatten()

            # Prepare input tensors for the model
            x = torch.FloatTensor(np.linspace(0, 1, len(scaled_timestamps))).reshape(-1, 1).to(device)
            t = torch.FloatTensor(scaled_timestamps).reshape(-1, 1).to(device)

            # Concatenate x and t for input to PINN
            xt = torch.cat([x, t], dim=1).to(device)  # This should give a shape of (N, 2)

            # Process with PINN model
            with torch.no_grad():
                denoised_data_scaled = pinn_model(xt).cpu().numpy().flatten()

            denoised_data = pinn_value_scaler.inverse_transform(denoised_data_scaled.reshape(-1, 1)).flatten()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scaling or processing data with PINN model: {str(e)}")

        # Step 2: GAN processing
        try:
            gan_input = gan_scaler.transform(denoised_data.reshape(-1, 1)).flatten()
            gan_sequences = []
            for i in range(0, len(gan_input) - 100 + 1, 1):
                gan_sequences.append(gan_input[i:i+100])
            gan_sequences = torch.FloatTensor(gan_sequences).to(device)

            # Generate samples using GAN
            num_samples = 10
            latent_dim = 100
            generated_samples = generate_samples(gan_generator, num_samples, latent_dim, device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing data for GAN model: {str(e)}")

        # Prepare augmented data for VAE
        augmented_data = np.concatenate([gan_sequences.cpu().numpy(), generated_samples.cpu().numpy()])

        # Step 3: VAE processing for anomaly detection
        try:
            vae_input = torch.FloatTensor(augmented_data).to(device)
            anomalies, original_data, reconstructed_data = detect_anomalies(vae_model, vae_input, device, vae_scaler)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error detecting anomalies using VAE model: {str(e)}")

        # Prepare results
        results = {
            "denoised_data": denoised_data.tolist(),
            "generated_samples": gan_scaler.inverse_transform(generated_samples).tolist(),
            "anomalies": anomalies.tolist(),
            "reconstructed_data": vae_scaler.inverse_transform(reconstructed_data).tolist()
        }

        logger.info("Seismic data processed successfully")
        return results

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing seismic data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the Seismic Detection API"}
