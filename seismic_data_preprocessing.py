import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler

def preprocess_seismic_data(data):
    """
    Preprocess seismic data by filtering noise and normalizing the signal.
    :param data: numpy array of raw seismic data
    :return: preprocessed data
    """
    # Butterworth filter for noise reduction
    b, a = signal.butter(3, 0.05)
    filtered_data = signal.filtfilt(b, a, data)

    # Normalize data between 0 and 1
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(filtered_data.reshape(-1, 1))
    
    return normalized_data

if __name__ == "__main__":
    raw_data = np.random.randn(1000)  # Simulated raw seismic data
    processed_data = preprocess_seismic_data(raw_data)
    print("Processed Data:", processed_data)
