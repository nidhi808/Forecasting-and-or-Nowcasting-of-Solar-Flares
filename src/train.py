import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix

# Ensure root dir is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mock_generator import generate_flare_profile
from src.data.cadence_sync import sync_cadences
from src.data.noise_filter import apply_savgol_filter
from src.nowcasting.nowcaster import detect_flares_rule_based, MLNowcaster
from src.forecasting.forecaster import (
    engineer_forecasting_features,
    create_labels,
    XGBForecaster,
    CNNLSTMForecasterWrapper,
    prepare_sequence_data
)

def create_nowcast_labels(df, events):
    y = np.zeros(len(df), dtype=int)
    if events is None or events.empty:
        return y
        
    # Sort events by peak_flux ascending so larger flares overwrite smaller ones
    sorted_events = events.copy()
    if 'peak_flux' in sorted_events.columns:
        sorted_events = sorted_events.sort_values('peak_flux')
        
    for _, flare in sorted_events.iterrows():
        start = flare['start_time']
        end = flare['end_time']
        gc = str(flare['goes_class'])
        
        if 'X-Class' in gc:
            lbl = 3
        elif 'M-Class' in gc:
            lbl = 2
        elif 'C-Class' in gc:
            lbl = 1
        else:
            lbl = 0 # quiet / Sub-flare / B-class
            
        if lbl > 0:
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            y[mask] = lbl
            
    return y

def run_training_pipeline():
    print("--------------------------------------------------")
    print("Starting Solar Flare Model Training Pipeline...")
    print("--------------------------------------------------")
    
    # 1. Generate 2 days of mock telemetry data to ensure robust training size
    print("Generating 2 days of high-frequency mock solar telemetry...")
    duration_seconds = 86400 * 2
    start_time = datetime.fromisoformat("2026-07-03T00:00:00")
    t_sxr = np.arange(0, duration_seconds, 1)
    t_hxr = np.arange(0, duration_seconds, 2)
    
    bg_sxr = 1.2e-7 + np.random.normal(0, 1e-9, len(t_sxr)) + 2e-8 * np.sin(2 * np.pi * t_sxr / 43200)
    bg_hxr = 2.0e-8 + np.random.normal(0, 5e-10, len(t_hxr)) + 3e-9 * np.sin(2 * np.pi * t_hxr / 43200)
    
    # Inject multiple flares of varying sizes
    flares_schedule = [
        {"peak_t": 25000, "peak_flux_sxr": 4e-6, "peak_flux_hxr": 8e-7, "rise": 400, "decay": 1500},
        {"peak_t": 60000, "peak_flux_sxr": 3e-5, "peak_flux_hxr": 8e-6, "rise": 600, "decay": 2500},
        {"peak_t": 110000, "peak_flux_sxr": 1.5e-4, "peak_flux_hxr": 5e-5, "rise": 800, "decay": 4000},
        {"peak_t": 150000, "peak_flux_sxr": 6e-6, "peak_flux_hxr": 1.2e-6, "rise": 350, "decay": 1200}
    ]
    
    sxr_flare_flux = np.zeros_like(t_sxr, dtype=float)
    hxr_flare_flux = np.zeros_like(t_hxr, dtype=float)
    
    for flare in flares_schedule:
        sxr_flare_flux += generate_flare_profile(t_sxr, flare["peak_t"], flare["peak_flux_sxr"], flare["rise"], flare["decay"])
        hxr_flare_flux += generate_flare_profile(t_hxr, flare["peak_t"] - 60, flare["peak_flux_hxr"], flare["rise"] // 2, flare["decay"] // 5)
        
    flux_sxr = bg_sxr + sxr_flare_flux
    flux_hxr = bg_hxr + hxr_flare_flux
    
    # Inject cosmic ray spikes
    np.random.seed(42)
    flux_sxr[np.random.choice(len(t_sxr), 800, replace=False)] += np.random.uniform(5e-6, 5e-5, 800)
    flux_hxr[np.random.choice(len(t_hxr), 500, replace=False)] += np.random.uniform(1e-6, 1.5e-5, 500)
    
    times_sxr = [start_time + timedelta(seconds=int(sec)) for sec in t_sxr]
    times_hxr = [start_time + timedelta(seconds=int(sec)) for sec in t_hxr]
    
    df_sxr = pd.DataFrame({"timestamp": times_sxr, "flux_solexs": flux_sxr})
    df_hxr = pd.DataFrame({"timestamp": times_hxr, "flux_hel1os": flux_hxr})
    
    # 2. Sync Cadences & Clean Data
    print("Synchronizing instruments & cleaning noise spikes...")
    df_sync = sync_cadences(df_sxr, df_hxr)
    df_sync['flux_solexs_clean'] = apply_savgol_filter(df_sync['flux_solexs'], window_length=31)
    df_sync['flux_hel1os_clean'] = apply_savgol_filter(df_sync['flux_hel1os'], window_length=15)
    
    # 3. Detect events and create labels
    print("Nowcasting rule-based event detection...")
    events = detect_flares_rule_based(df_sync)
    print(f"Found {len(events)} flares using rule-based nowcaster.")
    
    nowcast_labels = create_nowcast_labels(df_sync, events)
    forecasting_labels = create_labels(df_sync, events)
    
    # 4. Feature Engineering
    print("Engineering features...")
    df_feat, feature_names = engineer_forecasting_features(df_sync)
    
    # Chronological Split (70% train, 30% test) to prevent temporal data leakage
    split_idx = int(0.7 * len(df_sync))
    
    # --- Nowcasting Model Training (Random Forest) ---
    print("\n--- 1. Training Random Forest MLNowcaster ---")
    nowcaster = MLNowcaster()
    
    # Extract features for nowcaster
    X_now = nowcaster.extract_features(df_sync)
    X_now_train, X_now_test = X_now[:split_idx], X_now[split_idx:]
    y_now_train, y_now_test = nowcast_labels[:split_idx], nowcast_labels[split_idx:]
    
    # Train
    nowcaster.train(df_sync.iloc[:split_idx].reset_index(drop=True), y_now_train, model_path='models/nowcaster_model.joblib')
    
    # Evaluate
    y_now_pred = nowcaster.model.predict(X_now_test)
    unique_labels = np.unique(y_now_test)
    class_names = ['Quiet', 'C-Class', 'M-Class', 'X-Class']
    target_names = [class_names[i] for i in unique_labels]
    
    print("\nMLNowcaster Classification Report:")
    print(classification_report(y_now_test, y_now_pred, target_names=target_names))
    
    # --- Forecasting Model 1: XGBoost ---
    print("\n--- 2. Training XGBoost Forecaster ---")
    X_fore = df_feat[feature_names].values
    y_fore = forecasting_labels
    
    X_fore_train, X_fore_test = X_fore[:split_idx], X_fore[split_idx:]
    y_fore_train, y_fore_test = y_fore[:split_idx], y_fore[split_idx:]
    
    xgb_forecaster = XGBForecaster()
    
    # Balance class weights due to high class imbalance
    neg_count = sum(y_fore_train == 0)
    pos_count = sum(y_fore_train == 1)
    xgb_forecaster.model.scale_pos_weight = neg_count / pos_count if pos_count > 0 else 10.0
    
    xgb_forecaster.train(X_fore_train, y_fore_train, model_path='models/xgboost_forecaster.json')
    
    # Evaluate XGBoost
    y_fore_pred_xgb = (xgb_forecaster.predict_probability(X_fore_test) > 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_fore_test, y_fore_pred_xgb).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sensitivity + specificity - 1
    
    print("\nXGBoost Forecaster Classification Report:")
    print(classification_report(y_fore_test, y_fore_pred_xgb))
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"TSS: {tss:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    # --- Forecasting Model 2: PyTorch 1D CNN-LSTM ---
    print("\n--- 3. Training PyTorch 1D CNN-LSTM Forecaster ---")
    cnn_lstm_forecaster = CNNLSTMForecasterWrapper(input_dim=len(feature_names), seq_len=60)
    cnn_lstm_forecaster.train(X_fore_train, y_fore_train, epochs=10, model_path='models/cnn_lstm_model.pt')
    
    # Evaluate CNN-LSTM
    X_fore_test_scaled = cnn_lstm_forecaster.scaler.transform(X_fore_test)
    X_seq_test, y_seq_test = prepare_sequence_data(X_fore_test_scaled, y_fore_test, seq_len=60)
    
    if len(X_seq_test) > 0:
        cnn_lstm_forecaster.model.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(cnn_lstm_forecaster.device)
            preds_prob = cnn_lstm_forecaster.model(seq_tensor).cpu().numpy()
            
        preds_bin = (preds_prob > 0.5).astype(int)
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_seq_test, preds_bin).ravel()
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
        tss_c = sens_c + spec_c - 1
        
        print("\nPyTorch CNN-LSTM Forecaster Classification Report:")
        print(classification_report(y_seq_test, preds_bin))
        print(f"Confusion Matrix: TN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}")
        print(f"TSS: {tss_c:.4f}, Sensitivity: {sens_c:.4f}, Specificity: {spec_c:.4f}")
        
    print("\n--------------------------------------------------")
    print("Training Pipeline Completed Successfully!")
    print("Model files generated in the models/ directory.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_training_pipeline()
