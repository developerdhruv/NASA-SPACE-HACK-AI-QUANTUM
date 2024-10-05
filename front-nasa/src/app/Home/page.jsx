'use client'
import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Page = () => {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/process_seismic_file/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setData(response.data);
    } catch (err) {
      setError('Error processing file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const processGraphData = (dataArray, label) => {
    return dataArray.map((value, index) => ({
      index,
      [label]: value,
    }));
  };

  return (
    <div className="w-full min-h-screen bg-black text-white">
      <nav className="w-full h-16 bg-none flex items-center justify-between px-4">
        <div className="flex items-center">
          <div className="w-[40px] h-[20px] mr-2"></div>
          <h1 className="text-4xl text-white font-bold">Seismic DetAI</h1>
        </div>
        <div className="flex-grow flex justify-center">
          <div className="flex space-x-4 text-white">
            <a href="#home" className="hover:text-gray-300">Home</a>
            <a href="#about" className="hover:text-gray-300">About Us</a>
            <a href="#how-it-works" className="hover:text-gray-300">How It Works</a>
          </div>
        </div>
      </nav>
      
      <div className="w-full min-h-[80vh] bg-white p-8">
        <h1 className="text-5xl font-bold font-mono text-black mb-6 text-center">Upload The CSV Here</h1>
        <div className="flex justify-center items-center mb-4">
          <label className="h-10 bg-[#0080ff] rounded-[10px] px-4 py-2 text-white cursor-pointer mr-4">
            <input type='file' onChange={handleFileChange} className="hidden" />
            Choose File
          </label>
          <button 
            className="h-10 bg-[#0080ff] rounded-xl text-white font-bold px-4"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Submit'}
          </button>
        </div>

        {error && <p className="text-red-500 mt-4 text-center">{error}</p>}

        {data && (
          <div className="grid grid-cols-1 gap-8 mt-8">
            <div className="bg-gray-100 p-4 rounded shadow">
              <h2 className="text-2xl font-bold mb-4 text-black text-center">Denoised Data</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={processGraphData(data.denoised_data, 'Denoised')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Time', position: 'bottom', offset: 0 }} />
                  <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Denoised" stroke="#8884d8" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-100 p-4 rounded shadow">
              <h2 className="text-2xl font-bold mb-4 text-black text-center">Generated Samples</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={processGraphData(data.generated_samples[0], 'Generated')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Time', position: 'bottom', offset: 0 }} />
                  <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Generated" stroke="#82ca9d" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-100 p-4 rounded shadow">
              <h2 className="text-2xl font-bold mb-4 text-black text-center">Anomalies</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={processGraphData(data.anomalies.map(v => v ? 1 : 0), 'Anomaly')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Time', position: 'bottom', offset: 0 }} />
                  <YAxis domain={[0, 1]} label={{ value: 'Anomaly', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="step" dataKey="Anomaly" stroke="#ffc658" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-100 p-4 rounded shadow">
              <h2 className="text-2xl font-bold mb-4 text-black text-center">Reconstructed Data</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={processGraphData(data.reconstructed_data[0], 'Reconstructed')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Time', position: 'bottom', offset: 0 }} />
                  <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Reconstructed" stroke="#ff7300" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Page;