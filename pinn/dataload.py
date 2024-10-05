import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_seismic_data(file_path, save_scalers=True, scaler_save_path="scalers/"):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print the columns to debug
    print("Columns in the DataFrame:", df.columns)
    
    # Check if required columns exist
    required_columns = ['Time', 'Value', 'Network', 'Station', 'Location', 'Channel']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"The DataFrame does not contain a '{col}' column.")
    
    # Convert the 'Time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Calculate the time in seconds from the start
    df['TimeSeconds'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
    
    # Normalize time to [0, 1] range
    time_scaler = MinMaxScaler()
    time_normalized = time_scaler.fit_transform(df['TimeSeconds'].values.reshape(-1, 1))
    
    # Normalize Value to [-1, 1] range
    value_scaler = MinMaxScaler(feature_range=(-1, 1))
    value_normalized = value_scaler.fit_transform(df['Value'].values.reshape(-1, 1))
    
    # Convert to PyTorch tensors
    t = torch.FloatTensor(time_normalized)
    u = torch.FloatTensor(value_normalized)
    
    # Create a position vector (assuming 1D spatial dimension)
    x = torch.linspace(0, 1, len(t)).reshape(-1, 1)
    
    # Optionally save the scalers
    if save_scalers:
        joblib.dump(value_scaler, f'{scaler_save_path}value_scaler.pkl')
        joblib.dump(time_scaler, f'{scaler_save_path}time_scaler.pkl')
        print(f"Scalers saved to {scaler_save_path}")
    
    # Return additional metadata
    metadata = {
        'Network': df['Network'].iloc[0],
        'Station': df['Station'].iloc[0],
        'Location': df['Location'].iloc[0],
        'Channel': df['Channel'].iloc[0]
    }
    
    return x, t, u, value_scaler, metadata

def prepare_training_data(x, t, u, noise_level=0.05):
    # Add noise to the original signal
    u_noisy = u + noise_level * torch.randn_like(u)
    
    # Combine x and t for input
    xt = torch.cat([x, t], dim=1)
    
    return xt, u, u_noisy