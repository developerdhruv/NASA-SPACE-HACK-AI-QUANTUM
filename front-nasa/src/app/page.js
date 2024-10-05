'use client'
import React from 'react';
import './globals.css';
import { useRouter } from 'next/navigation'; // Import useRouter
const Page = () => {
  const rounter = useRouter();
  const nav= (e) => {
    e.preventDefault();
    
    rounter.push('/Register');
  }
  return (
    <>
      <div className="w-full h-full bg-none main">
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
          <div>
            <button className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 " onClick={nav}>
              Login
            </button>
          </div>
        </nav>
        <div className = "W-full h-[100vh] grid justify-center content-center">
          <h1 className = "text-8xl font-bold font-mono">Seismic DETAI</h1>
          <h2 className = "text-3xl font-bold font-mono align-center pl-[70px]">Unlocking Secrets of Space with </h2>
          <h2 className = "text-3xl font-bold font-mono align-center pl-22 pl-[90px]">AI Enabled Physics Platform </h2>
          <button className = " h-10 bg-[#0080ff] rounded-xl   " onClick={nav}>
            <h1 className = "text-xl font-bold">Get Started</h1>
            </button>

        </div>
      </div>
    </>
  );
};

export default Page;
