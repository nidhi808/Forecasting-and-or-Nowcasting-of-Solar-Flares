import os
import sys
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fits_processor import read_l1_fits
from src.data.cadence_sync import sync_cadences
from src.data.noise_filter import clean_time_series
from src.nowcasting.nowcaster import detect_flares_rule_based, MLNowcaster
from src.forecasting.forecaster import engineer_forecasting_features, XGBForecaster

app = FastAPI(title="Solar Flare Nowcasting & Forecasting Telemetry API")

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store live telemetry and models
live_telemetry_df = None
master_catalog_df = None
xgb_forecaster = None
ml_nowcaster = None

# Cache directory paths
DATA_DIR = "data/raw"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize models
def load_models():
    global xgb_forecaster, ml_nowcaster
    
    # Load XGBoost forecaster
    xgb_forecaster = XGBForecaster()
    xgb_path = os.path.join(MODELS_DIR, "xgboost_forecaster.json")
    if os.path.exists(xgb_path):
        try:
            xgb_forecaster.load(xgb_path)
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
    else:
        print("XGBoost forecaster model not found. Running in simulation mode for forecasting.")
        
    # Load ML nowcaster (Random Forest)
    ml_nowcaster = MLNowcaster()
    nowcast_path = os.path.join(MODELS_DIR, "nowcaster_model.joblib")
    if os.path.exists(nowcast_path):
        try:
            ml_nowcaster.load(nowcast_path)
        except Exception as e:
            print(f"Error loading ML nowcaster: {e}")
    else:
        print("ML nowcaster model not found. Using rule-based nowcasting.")

# Seed or load default mock data
def load_initial_data():
    global live_telemetry_df, master_catalog_df
    
    # Check if mock FITS files exist, if not generate them
    solexs_fits = os.path.join(DATA_DIR, "aditya_solexs_level1_20260703.fits")
    hel1os_fits = os.path.join(DATA_DIR, "aditya_hel1os_level1_20260703.fits")
    
    if not os.path.exists(solexs_fits) or not os.path.exists(hel1os_fits):
        print("Generating mock FITS data...")
        from src.data.mock_generator import generate_mock_data
        generate_mock_data(output_dir=DATA_DIR)
        
    # Process and sync FITS files
    try:
        df_sxr, _ = read_l1_fits(solexs_fits)
        df_hxr, _ = read_l1_fits(hel1os_fits)
        
        # Sync
        synced = sync_cadences(df_sxr, df_hxr)
        
        # Filter
        cleaned = clean_time_series(synced)
        live_telemetry_df = cleaned
        
        # Run rule-based nowcast for the master catalog
        master_catalog_df = detect_flares_rule_based(cleaned)
        print(f"Data initialized. Telemetry size: {len(live_telemetry_df)}, Catalog size: {len(master_catalog_df)}")
    except Exception as e:
        print(f"Error initializing data: {e}")
        # Fallback empty df
        live_telemetry_df = pd.DataFrame()
        master_catalog_df = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    load_models()
    load_initial_data()

# API Endpoints
class TelemetryRecord(BaseModel):
    timestamp: str
    flux_solexs: float
    flux_hel1os: float
    flux_solexs_clean: float
    flux_hel1os_clean: float

@app.get("/api/telemetry/live")
def get_live_telemetry(limit: int = 1200):
    """
    Returns the latest N seconds of telemetry.
    """
    global live_telemetry_df
    if live_telemetry_df is None or live_telemetry_df.empty:
        return []
        
    # Take the last N records
    latest = live_telemetry_df.tail(limit).copy()
    latest['timestamp'] = latest['timestamp'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Replace NaNs with None for JSON serialization
    latest = latest.replace({np.nan: None})
    return latest.to_dict(orient="records")

@app.get("/api/nowcast/catalog")
def get_nowcast_catalog():
    """
    Returns the master catalog of all detected solar flares.
    """
    global master_catalog_df
    if master_catalog_df is None or master_catalog_df.empty:
        return []
        
    catalog = master_catalog_df.copy()
    catalog['start_time'] = catalog['start_time'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog['peak_time'] = catalog['peak_time'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog['end_time'] = catalog['end_time'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return catalog.to_dict(orient="records")

@app.get("/api/predict")
def get_live_forecast_probability():
    """
    Calculates and returns the probability of a solar flare in the next 15-45 mins
    using the latest 60 minutes of telemetry.
    """
    global live_telemetry_df, xgb_forecaster
    if live_telemetry_df is None or len(live_telemetry_df) < 3600:
        # Fallback to simulation probability if not enough data
        import random
        # Simulate high probability if we're near the flares
        # In mock data, peaks are at t=15000, 45000, 70000
        # We can simulate warning state
        return {"probability": round(random.uniform(0.01, 0.05), 3), "status": "Quiet"}
        
    # Get latest 60 minutes (3600 seconds)
    latest_df = live_telemetry_df.tail(3600).reset_index(drop=True)
    
    # Engineer features
    feat_df, feat_cols = engineer_forecasting_features(latest_df)
    X = feat_df[feat_cols].tail(1).values
    
    if xgb_forecaster and xgb_forecaster.is_trained:
        prob = xgb_forecaster.predict_probability(X)[0]
    else:
        # Simulating based on Clean SoLEXS derivative velocity
        # If soft x-ray starts rising rapidly, probability shoots up
        last_sxr_clean = latest_df['flux_solexs_clean'].iloc[-1]
        last_deriv = np.gradient(latest_df['flux_solexs_clean'].values)[-1]
        
        # High derivative = high forecast probability
        if last_deriv > 1e-8:
            prob = min(0.95, 0.4 + last_deriv * 1e7)
        elif last_deriv > 2e-9:
            prob = min(0.60, 0.1 + last_deriv * 1e7)
        else:
            prob = max(0.01, 0.05 + last_deriv * 1e7)
            
    status = "Quiet"
    if prob > 0.8:
        status = "CRITICAL WARNING"
    elif prob > 0.5:
        status = "WARNING"
    elif prob > 0.2:
        status = "ELEVATED PRECURSOR ACTIVE"
        
    return {
        "probability": float(round(prob, 3)),
        "status": status,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }

@app.post("/api/upload")
async def upload_fits_files(solexs: UploadFile = File(...), hel1os: UploadFile = File(...)):
    """
    Accepts uploaded SoLEXS and HEL1OS Level-1 FITS files, processes them, 
    detects flares, and returns the list of events found in the files.
    """
    # Create temp files
    temp_solexs_path = f"data/raw/temp_{solexs.filename}"
    temp_hel1os_path = f"data/raw/temp_{hel1os.filename}"
    
    try:
        with open(temp_solexs_path, "wb") as f:
            f.write(await solexs.read())
        with open(temp_hel1os_path, "wb") as f:
            f.write(await hel1os.read())
            
        # Process FITS
        df_sxr, _ = read_l1_fits(temp_solexs_path)
        df_hxr, _ = read_l1_fits(temp_hel1os_path)
        
        # Sync
        synced = sync_cadences(df_sxr, df_hxr)
        
        # Filter
        cleaned = clean_time_series(synced)
        
        # Nowcast
        detected_flares = detect_flares_rule_based(cleaned)
        
        # Serialize timestamps
        events = []
        if not detected_flares.empty:
            for _, flare in detected_flares.iterrows():
                events.append({
                    "instrument": flare["instrument"],
                    "start_time": flare["start_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "peak_time": flare["peak_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_time": flare["end_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "peak_flux": flare["peak_flux"],
                    "goes_class": flare["goes_class"]
                })
                
        return {
            "status": "success",
            "message": "FITS files processed successfully",
            "flares_detected_count": len(events),
            "events": events
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_solexs_path):
            os.remove(temp_solexs_path)
        if os.path.exists(temp_hel1os_path):
            os.remove(temp_hel1os_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
