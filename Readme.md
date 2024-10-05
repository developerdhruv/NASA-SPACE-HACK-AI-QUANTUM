# NASA Space Apps Challenge 2024 [Noida]

#### Team Name -CloudIspy
#### Problem Statement - Seismic Detection across Solar system
#### Team Leader Email - dhruvkumar9115@gmail.com

## A Brief of the Prototype:
# Seismic Data Processing API

## Project Overview

This project is a **seismic data processing prototype** . It performs advanced seismic data processing through the integration of multiple deep learning models and visulise them on a nextjs based web app:

1. **Denoising**: Reduces noise in seismic data using a **Physics-Informed Neural Network (PINN)**.
2. **Sample Generation**: Generates synthetic seismic data samples using a **Generative Adversarial Network (GAN)**.
3. **Anomaly Detection**: Detects anomalies using a **Variational Autoencoder (VAE)**.

The Platform is designed to be lightweight, scalable, and easy to deploy in cloud environments, making it an ideal solution for real-time seismic monitoring and data analysis.


---

## Execution Plan

### 1. **Data Preprocessing**
   - **Input**: Seismic data in CSV format containing values and optional timestamps.
   - **Preprocessing**: Seismic data is normalized and scaled using MinMaxScaler for further processing by the models.

### 2. **Denoising with PINN**
   - The **PINN** model denoises the input seismic data by learning the underlying physical properties of seismic waves.
   - **Output**: Clean, denoised seismic data.

### 3. **Sample Generation with GAN**
   - The **GAN** model generates synthetic samples to augment the seismic data.
   - **Output**: Generated seismic data samples that mimic real seismic events.

### 4. **Anomaly Detection with VAE**
   - The **VAE** model detects anomalies by comparing the reconstructed seismic data to the original input.
   - **Output**: Anomalies detected in the seismic data, highlighting potential risks.
## YouTube Video Demo

For a quick walkthrough and demo of the API and Frontend in action, check out our [YouTube Video Demo](https://youtu.be/ALaTSZ6iiIM).

---

 **API Endpoints**:
    - The main API endpoint for seismic data processing:
      ```http
      POST /process_seismic_file/
      ```
    - Example request:
      Upload a CSV file containing 'Timestamp' and 'Value' columns.

 **Test the API**:
   - You can test the API using tools like **Postman** or **curl**:
     ```bash
     curl -X POST "http://localhost:8000/process_seismic_file/" -F "file=@seismic_data.csv"
     ```

---


## Contact

For any questions, feel free to reach out to the project maintainer:

- Name: Dhruv
- Email: dhruv@example.com