Medicine Authenticity Checker - Setup Guide

Prerequisites

Node.js (v14+)

Python (v3.8+)

Expo CLI (npm install -g expo-cli)

Installation

1. Backend Setup

Navigate to the auth-checker root directory.

cd backend
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install fastapi uvicorn pydantic python-multipart requests


2. Frontend Setup

cd ../app
npm install
npm install expo-image-picker expo-status-bar


Running the Project

Start the Backend:
Inside backend/:

python server.py


The API will run at http://localhost:8000.

Start the App:
Inside app/:

npx expo start


Scan the QR code with your physical device (Expo Go app required) or use a simulator.

Architecture

Mobile App: React Native (Expo). Handles image capture and displays results.

Backend: Python FastAPI. Acts as the gateway.

ML Engine: Uses simulated OCR and Visual Analysis to determine a confidence score (0-100%).