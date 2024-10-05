import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
df = pd.read_csv('../seiss_data.csv')
def process_data(df, save_scalers=False, scaler_save_path='./'):
    # Check if 'Time' column exists
    if 'Time' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Time' column.")
    
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Calculate the time in seconds from the start
    df['TimeSeconds'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
    
    # Normalize time to [0, 1] range
    print("pass")

def prepare_training_data(x, t, u, noise_level=0.05):
    # Your implementation for prepare_training_data
    pass


process_data(df=df, save_scalers=False, scaler_save_path='./')