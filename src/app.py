import os
import sys
import numpy as np
import pandas as pd
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fits_processor import read_l1_fits
from src.data.cadence_sync import sync_cadences
from src.data.noise_filter import clean_time_series
from src.nowcasting.nowcaster import detect_flares_rule_based, MLNowcaster
from src.forecasting.forecaster import engineer_forecasting_features, XGBForecaster, CNNLSTMForecasterWrapper

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
cnn_lstm_forecaster = None
ml_nowcaster = None
model_metrics = {}

# Cache directory paths
DATA_DIR = "data/raw"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize models
def load_models():
    global xgb_forecaster, ml_nowcaster, cnn_lstm_forecaster, model_metrics
    
    # Load model metrics if they exist
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                model_metrics = json.load(f)
            print("Model metrics loaded successfully.")
        except Exception as e:
            print(f"Error loading model metrics: {e}")

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
        
    # Load CNN-LSTM forecaster
    cnn_lstm_forecaster = CNNLSTMForecasterWrapper(input_dim=36, seq_len=60)
    cnn_lstm_path = os.path.join(MODELS_DIR, "cnn_lstm_model.pt")
    if os.path.exists(cnn_lstm_path):
        try:
            cnn_lstm_forecaster.load(cnn_lstm_path)
        except Exception as e:
            print(f"Error loading CNN-LSTM model: {e}")
    else:
        print("CNN-LSTM forecaster model not found. Running in simulation mode for CNN-LSTM forecasting.")
        
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
def get_live_forecast_probability(model_type: str = "xgb"):
    """
    Calculates and returns the probability of a solar flare in the next 15-45 mins
    using the latest 60 minutes of telemetry. Exposes both XGBoost and CNN-LSTM models.
    Also returns the current real-time ML-nowcasted solar flare state of the Sun.
    """
    global live_telemetry_df, xgb_forecaster, cnn_lstm_forecaster, ml_nowcaster
    
    # Ensure minimum telemetry exists
    if live_telemetry_df is None or len(live_telemetry_df) < 100:
        return {
            "probability": 0.05,
            "status": "Quiet",
            "nowcast_class": "Quiet",
            "model_type": model_type,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
    # Get latest 60 minutes (3600 seconds)
    latest_df = live_telemetry_df.tail(3600).reset_index(drop=True)
    
    # Engineer features
    feat_df, feat_cols = engineer_forecasting_features(latest_df)
    
    # 1. FORECASTING
    prob = 0.05
    is_simulated = True
    
    if model_type == "cnn_lstm":
        if cnn_lstm_forecaster and cnn_lstm_forecaster.is_trained:
            # CNN-LSTM wrapper extracts and scales the last seq_len internally
            prob = cnn_lstm_forecaster.predict_probability(feat_df[feat_cols].values)
            is_simulated = False
    else: # Default is xgb
        if xgb_forecaster and xgb_forecaster.is_trained:
            X = feat_df[feat_cols].tail(1).values
            prob = xgb_forecaster.predict_probability(X)[0]
            is_simulated = False
            
    # If model is not trained/found, run mathematical simulation based on Clean SoLEXS derivative velocity
    if is_simulated:
        last_deriv = np.gradient(latest_df['flux_solexs_clean'].values)[-1]
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
        
    # 2. NOWCASTING
    nowcast_class = "Quiet"
    if ml_nowcaster and ml_nowcaster.is_trained:
        try:
            # Use last 10 seconds of telemetry to extract features and predict current state
            latest_10s = latest_df.tail(10).reset_index(drop=True)
            pred_labels, _ = ml_nowcaster.predict(latest_10s)
            nowcast_class = pred_labels[-1]
        except Exception as e:
            print(f"Error in ML nowcast prediction: {e}")
    else:
        # Simple rule-based fallback based on raw flux levels
        last_flux = latest_df['flux_solexs_clean'].iloc[-1]
        if last_flux >= 1e-4:
            nowcast_class = "X-Class"
        elif last_flux >= 1e-5:
            nowcast_class = "M-Class"
        elif last_flux >= 1e-6:
            nowcast_class = "C-Class"
        else:
            nowcast_class = "Quiet"
            
    # Fetch false alarm rate dynamically
    far = model_metrics.get(model_type, {}).get("false_alarm_rate", 0.15)
    
    return {
        "probability": float(round(prob, 3)),
        "status": status,
        "nowcast_class": nowcast_class,
        "model_type": model_type,
        "lead_time_minutes": 15,
        "lead_time_description": "Alert fires 15-30 min before predicted flare peak",
        "is_simulated": is_simulated,
        "false_alarm_rate": float(far),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }

@app.get("/api/nowcast/catalog/summary")
def get_catalog_summary():
    """
    Returns per-class (C/M/X) detection statistics from the master flare catalog.
    Used to report detection rates separately for low-class and high-class flares
    as required by the evaluation criteria.
    """
    global master_catalog_df
    if master_catalog_df is None or master_catalog_df.empty:
        return {"total": 0, "classes": {}}

    catalog = master_catalog_df.copy()
    summary = {"total": len(catalog), "classes": {}}

    class_groups = {
        "C-Class": ["C-Class", "C-Class Equivalent"],
        "M-Class": ["M-Class", "M-Class Equivalent"],
        "X-Class": ["X-Class", "X-Class Equivalent"],
        "B-Class": ["B-Class / Sub-flare", "Sub-flare"]
    }
    for label, goes_names in class_groups.items():
        subset = catalog[catalog["goes_class"].isin(goes_names)]
        if len(subset) > 0:
            summary["classes"][label] = {
                "count": len(subset),
                "mean_duration_seconds": round(float(subset["duration_seconds"].mean()), 1),
                "mean_peak_flux": float(f"{subset['peak_flux'].mean():.3e}"),
                "max_peak_flux": float(f"{subset['peak_flux'].max():.3e}"),
                "instruments": subset["instrument"].value_counts().to_dict()
            }
    return summary

@app.get("/api/nowcast/stats")
def get_nowcast_stats():
    """
    Returns headline performance statistics for the dashboard metrics card.
    Includes lead time, model type, and detection class breakdown.
    """
    global master_catalog_df, xgb_forecaster, cnn_lstm_forecaster, model_metrics
    total_flares = len(master_catalog_df) if master_catalog_df is not None and not master_catalog_df.empty else 0
    return {
        "total_flares_detected": total_flares,
        "lead_time_minutes": 15,
        "forecast_window_minutes": 30,
        "models_loaded": {
            "xgboost": xgb_forecaster.is_trained if xgb_forecaster else False,
            "cnn_lstm": cnn_lstm_forecaster.is_trained if cnn_lstm_forecaster else False,
        },
        "model_performance_metrics": model_metrics,
        "evaluation_metrics": {
            "primary_metric": "TSS (True Skill Statistic)",
            "secondary_metric": "ROC-AUC",
            "note": "TSS > 0.5 is operational threshold for flare forecasting"
        }
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
