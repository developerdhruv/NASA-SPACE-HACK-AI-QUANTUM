import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
from datetime import datetime
import os
import io
import asyncio

# Import custom modules
from pinn.main import SeismicPINN
from pinn.dataload import prepare_training_data
from gans.train import Generator, generate_samples
from vaee.train import VAE, detect_anomalies

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"seismic_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths to checkpoint files
PinnPath = './pinn/seismic_pinn_checkpoint.pth'
GanPath = './gans/generator_model.pth'
VaePath = './vaee/vae_model.pth'

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize FastAPI app
app = FastAPI(title="Seismic Processing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response
class SeismicData(BaseModel):
    values: List[float] = Field(..., description="List of seismic values")
    timestamps: Optional[List[float]] = Field(..., description="List of timestamps (optional)")

class ProcessingResult(BaseModel):
    denoised_data: List[float] = Field(..., description="Denoised seismic data")
    generated_samples: List[List[float]] = Field(..., description="Generated samples from GAN")
    anomalies: List[bool] = Field(..., description="Anomaly detection results")
    reconstructed_data: List[List[float]] = Field(..., description="Reconstructed data from VAE")

# Global variables for models and scalers
pinn_model = None
gan_generator = None
vae_model = None
pinn_time_scaler = None
pinn_value_scaler = None
gan_scaler = None
vae_scaler = None

async def load_models():
    global pinn_model, gan_generator, vae_model, pinn_time_scaler, pinn_value_scaler, gan_scaler, vae_scaler
    try:
        pinn_model = SeismicPINN([2, 50, 50, 50, 1]).to(device)
        pinn_model.load_state_dict(torch.load(PinnPath, map_location=device)['model_state_dict'])
        pinn_model.eval()

        gan_generator = Generator(input_dim=100, output_dim=100).to(device)
        gan_generator.load_state_dict(torch.load(GanPath, map_location=device))
        gan_generator.eval()

        vae_model = VAE(input_dim=50, hidden_dim=128, latent_dim=20).to(device)
        vae_model.load_state_dict(torch.load(VaePath, map_location=device))
        vae_model.eval()

        pinn_time_scaler = joblib.load('./pinn/scalers/time_scaler.pkl')
        pinn_value_scaler = joblib.load('./pinn/scalers/value_scaler.pkl')
        gan_scaler = joblib.load('./gans/gan_scaler.pkl')
        vae_scaler = joblib.load('./vaee/scaler.pkl')

        logger.info("All models and scalers loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models or scalers: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    await load_models()

def process_seismic_data_task(seismic_data: SeismicData) -> ProcessingResult:
    try:
        seismic_values = np.array(seismic_data.values)
        timestamps = np.array(seismic_data.timestamps) if seismic_data.timestamps else np.arange(len(seismic_values))

        # Step 1: PINN processing
        try:
            scaled_timestamps = pinn_time_scaler.transform(timestamps.reshape(-1, 1)).flatten()
            scaled_values = pinn_value_scaler.transform(seismic_values.reshape(-1, 1)).flatten()

            x = torch.FloatTensor(np.linspace(0, 1, len(scaled_timestamps))).reshape(-1, 1).to(device)
            t = torch.FloatTensor(scaled_timestamps).reshape(-1, 1).to(device)
            xt = torch.cat([x, t], dim=1).to(device)

            with torch.no_grad():
                denoised_data_scaled = pinn_model(xt).cpu().numpy().flatten()

            denoised_data = pinn_value_scaler.inverse_transform(denoised_data_scaled.reshape(-1, 1)).flatten()
            logger.info(f"PINN processing completed. Shape of denoised_data: {denoised_data.shape}")

        except Exception as e:
            logger.error(f"Error in PINN processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error in PINN processing: {str(e)}")

        # Step 2: GAN processing
        try:
            gan_input = gan_scaler.transform(denoised_data.reshape(-1, 1)).flatten()
            gan_sequences = []
            for i in range(0, len(gan_input) - 100 + 1, 100):
                gan_sequences.append(gan_input[i:i+100])
            gan_sequences = torch.FloatTensor(gan_sequences).to(device)

            num_samples = 10
            latent_dim = 100
            generated_samples = generate_samples(gan_generator, num_samples, latent_dim, device)
            logger.info(f"GAN processing completed. Shape of generated_samples: {generated_samples.shape}")

        except Exception as e:
            logger.error(f"Error in GAN processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error in GAN processing: {str(e)}")

        # Prepare augmented data for VAE
        try:
            if isinstance(generated_samples, torch.Tensor):
                generated_samples = generated_samples.cpu().numpy()
            augmented_data = np.concatenate([gan_sequences.cpu().numpy(), generated_samples])
            logger.info(f"Augmented data prepared. Shape: {augmented_data.shape}")

        except Exception as e:
            logger.error(f"Error in preparing augmented data: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error in preparing augmented data: {str(e)}")

        # Step 3: VAE processing for anomaly detection
        try:
            # Ensure the input is 2D: (batch_size, feature_size)
            if len(augmented_data.shape) == 1:
                augmented_data = augmented_data.reshape(1, -1)
            elif len(augmented_data.shape) > 2:
                augmented_data = augmented_data.reshape(-1, augmented_data.shape[-1])
            
            logger.info(f"Shape of augmented_data after reshaping: {augmented_data.shape}")
            
            vae_input = torch.FloatTensor(augmented_data).to(device)
            logger.info(f"Shape of vae_input: {vae_input.shape}")
            
            # Ensure the VAE model's input_dim matches the data
            if vae_input.shape[1] != vae_model.input_dim:
                logger.warning(f"Mismatch between VAE input_dim ({vae_model.input_dim}) and data shape ({vae_input.shape[1]})")
                # Adjust the input data to match the model's expected input size
                vae_input = vae_input[:, :vae_model.input_dim]
                logger.info(f"Adjusted vae_input shape: {vae_input.shape}")
            
            anomalies, original_data, reconstructed_data = detect_anomalies(vae_model, vae_input, device, vae_scaler)
            logger.info(f"VAE processing completed. Shape of reconstructed_data: {reconstructed_data.shape}")

        except Exception as e:
            logger.error(f"Error in VAE processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error in VAE processing: {str(e)}")

        # Prepare results
        try:
            if isinstance(generated_samples, np.ndarray):
                generated_samples_np = generated_samples
            else:
                generated_samples_np = generated_samples.cpu().numpy()

            if isinstance(reconstructed_data, np.ndarray):
                reconstructed_data_np = reconstructed_data
            else:
                reconstructed_data_np = reconstructed_data.cpu().numpy()

            results = ProcessingResult(
                denoised_data=denoised_data.tolist(),
                generated_samples=gan_scaler.inverse_transform(generated_samples_np).tolist(),
                anomalies=anomalies.tolist(),
                reconstructed_data=vae_scaler.inverse_transform(reconstructed_data_np).tolist()
            )
            logger.info("Results prepared successfully")

        except Exception as e:
            logger.error(f"Error in preparing results: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error in preparing results: {str(e)}")

        logger.info("Seismic data processed successfully")
        return results

    except Exception as e:
        logger.error(f"Unexpected error processing seismic data: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Unexpected error: {str(e)}")

@app.post("/process_seismic_file/", response_model=ProcessingResult)
async def process_seismic_file(file: UploadFile = File(...)):
    try:
        # Ensure that the file is in CSV format
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")

        # Read the contents of the file
        contents = await file.read()
        
        # Convert to pandas DataFrame
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Check for required columns
        if 'Timestamp' not in df.columns or 'Value' not in df.columns:
            raise ValueError("CSV file must contain both 'Timestamp' and 'Value' columns.")
        
        if df['Timestamp'].isnull().any() or df['Value'].isnull().any():
            raise ValueError("CSV file contains missing values in 'Timestamp' or 'Value' columns.")

        # Convert timestamps to float if they're not already
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        
        # Create SeismicData object
        seismic_data = SeismicData(
            values=df['Value'].tolist(),
            timestamps=df['Timestamp'].tolist()
        )
        
        # Process the data
        result = await asyncio.to_thread(process_seismic_data_task, seismic_data)
        return result
    except Exception as e:
        logger.error(f"Error in process_seismic_file endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Welcome to the Seismic Detection API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)