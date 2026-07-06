# Forecasting & Nowcasting of Solar Flares using Aditya-L1 X-ray Data

This repository contains the ground station heliophysics telemetry segment pipeline for automated nowcasting and forecasting of solar flares. It combines soft X-ray data from **SoLEXS** and hard X-ray data from **HEL1OS** instruments onboard ISRO's **Aditya-L1** spacecraft.

## 🚀 Features

* **Real-time Synchronization**: Resamples and aligns SoLEXS (1s cadence) and HEL1OS (2s cadence) timelines, handling orbital gaps and telemetry packet drops.
* **Noise Suppression**: Digital signal processing using linear interpolation and Savitzky-Golay filters to suppress high-frequency sensor noise and cosmic ray micro-spikes.
* **Algorithmic & ML Nowcasting**:
  * Algorithmic derivative-based event tracker that logs solar flares (start, peak, end, duration, and peak flux).
  * Machine learning Random Forest classifier classifying active solar states (`Quiet`, `C-Class`, `M-Class`, `X-Class`).
* **Advanced Predictive Forecasting**:
  * **XGBoost Classifier** to predict probability of a solar flare in the next 15-45 minutes (optimized for True Skill Statistic / TSS).
  * **PyTorch 1D CNN + LSTM Hybrid Network** modeling sequential momentum of solar X-ray flux variations.
* **Premium Space Weather Cockpit**: Interactive React dashboard with glassmorphic visuals, live Plotly light curve charts, warning alerts, model selection, and event logging tables.

---

## 📁 Codebase Directory Structure

```
├── dashboard/                  # React + Vite frontend dashboard SPA
│   ├── src/
│   │   ├── App.jsx             # Main SPA dashboard UI (Plotly chart, warning panel, FITS upload)
│   │   └── index.css           # Premium glassmorphism design system styles
│   └── package.json            # React project dependencies
│
├── data/                       # Local telemetry and satellite data storage
│   └── raw/                    # Simulated or real telemetry Level-1 FITS pairs
│
├── models/                     # Serialized machine learning models and scalers
│   ├── cnn_lstm_model.pt       # PyTorch CNN-LSTM forecaster weights
│   ├── cnn_lstm_model_scaler.joblib
│   ├── nowcaster_model.joblib  # Random Forest nowcasting model
│   ├── xgboost_forecaster.json # XGBoost forecaster weights
│   └── xgboost_forecaster_scaler.joblib
│
├── src/                        # Core Python Pipeline source modules
│   ├── app.py                  # FastAPI Backend API server
│   ├── train.py                # Standalone model training execution pipeline script
│   ├── data/
│   │   ├── fits_processor.py   # Astro FITS reader utilizing astropy.io.fits
│   │   ├── cadence_sync.py     # Alignment and resampling of instruments
│   │   ├── noise_filter.py     # Savitzky-Golay noise filters
│   │   └── mock_generator.py   # Synthetic Level-1 telemetry generator
│   ├── nowcasting/
│   │   └── nowcaster.py        # Algorithmic & ML nowcasting models
│   └── forecasting/
│       └── forecaster.py       # XGBoost and PyTorch CNN-LSTM forecasting models
│
├── tests/                      # Automated pipeline test suite
│   └── test_pipeline.py        # Pytest cases verifying DSP and ML load/predict
│
├── requirements.txt            # Python dependencies list
└── run_app.ps1                 # Powershell startup script for launching backend + frontend
```

---

## 🛠️ Installation & Setup

### Prerequisite
Ensure you have **Python 3.10+** and **Node.js 18+** installed.

### 1. Python Virtual Environment & Backend Setup
Open Powershell/Terminal in the project root:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend setup
Change directory into `dashboard/` and install dependencies:

```bash
cd dashboard
npm install
cd ..
```

---

## 🏋️ Model Training

### 1. Simulated Telemetry Training
To train the nowcasting and forecasting models on simulated Aditya-L1 telemetry (SoLEXS & HEL1OS cadences), run:

```bash
python src/train.py
```

### 2. Real GOES Satellite Telemetry Training (Recommended)
To train the models on 3 years of real NOAA GOES-16 XRS satellite observations (2023, 2024, and 2025):

1. **Download the 3-year dataset** (downloads 1 day per week of 1-minute averaged NetCDF files to `data/goes/`):
   ```bash
   python -m src.data.download_goes_3yr
   ```
2. **Train the models** (parses NetCDF files, cleans spikes, and retrains all models):
   ```bash
   python -m src.train_real_goes
   ```

---

## 🚀 Running the Space Weather Dashboard

You can start both the FastAPI backend server and React Vite frontend in dev mode by running the helper PowerShell script:

```powershell
./run_app.ps1
```

Or start them manually in separate shells:

**Start FastAPI Backend:**
```bash
python src/app.py
# Backend runs at http://127.0.0.1:8000
```

**Start React Dev Dashboard:**
```bash
cd dashboard
npm run dev
# Frontend dashboard URL will print in the console (usually http://localhost:5173)
```

---

## 🧪 Testing

The pipeline features unit tests covering Astro FITS reading, cadence synchronization, Savitzky-Golay noise filtering, event detection, and feature engineering.

Run the test suite using `pytest`:

```bash
pytest tests/
```