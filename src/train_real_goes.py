import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Ensure root dir is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.goes_parser import load_and_preprocess_goes_dataset
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

class PyTorchSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len=60):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]

def run_training_pipeline():
    print("--------------------------------------------------")
    print("Starting Solar Flare Real GOES Model Training Pipeline...")
    print("--------------------------------------------------")
    
    # 1. Load and parse real GOES NetCDF dataset (2023-2025)
    df_goes = load_and_preprocess_goes_dataset()
    if df_goes.empty:
        print("Error: GOES dataset is empty. Cannot proceed with training.")
        return
        
    # Rename columns to match SoLEXS/HEL1OS pipeline expectations
    df_goes.rename(columns={
        'time': 'timestamp',
        'flux_sxr': 'flux_solexs',
        'flux_hxr': 'flux_hel1os'
    }, inplace=True)
    
    # Map cleaned columns (already smoothed by goes_parser)
    df_goes['flux_solexs_clean'] = df_goes['flux_solexs']
    df_goes['flux_hel1os_clean'] = df_goes['flux_hel1os']
    
    # 2. Detect events and create labels
    # Using dt=60 because GOES-16 1-minute averaged data has a 60s cadence
    print("Nowcasting rule-based event detection (dt=60)...")
    events = detect_flares_rule_based(df_goes, dt=60)
    print(f"Found {len(events)} solar flares using rule-based nowcaster.")
    
    nowcast_labels = create_nowcast_labels(df_goes, events)
    forecasting_labels = create_labels(df_goes, events)
    
    # 3. Feature Engineering (cadence_seconds=60)
    print("Engineering features...")
    df_feat, feature_names = engineer_forecasting_features(df_goes, cadence_seconds=60)
    
    # Chronological Split (70% train, 30% test) to prevent temporal data leakage
    split_idx = int(0.7 * len(df_goes))
    
    # --- Nowcasting Model Training (Random Forest) ---
    print("\n--- 1. Training Random Forest MLNowcaster ---")
    nowcaster = MLNowcaster()
    
    # Extract features for nowcaster
    X_now = nowcaster.extract_features(df_goes)
    X_now_train, X_now_test = X_now[:split_idx], X_now[split_idx:]
    y_now_train, y_now_test = nowcast_labels[:split_idx], nowcast_labels[split_idx:]
    
    # Train
    nowcaster.train(df_goes.iloc[:split_idx].reset_index(drop=True), y_now_train, model_path='models/nowcaster_model.joblib')
    
    # Evaluate nowcaster
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
    
    # Balance class weights due to class imbalance
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
    # Use batch_size = 512 for faster CPU training on larger real dataset
    print("Training PyTorch model...")
    
    # Custom training loop to support larger batch size and faster epoch steps
    X_train_scaled = cnn_lstm_forecaster.scaler.fit_transform(X_fore_train)
    dataset = PyTorchSequenceDataset(X_train_scaled, y_fore_train, seq_len=60)
    # Batch size 2048 is fast on CPU for 1.2M rows
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(cnn_lstm_forecaster.model.parameters(), lr=0.001)
    
    cnn_lstm_forecaster.model.train()
    epochs = 3
    num_sequences = len(dataset)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(cnn_lstm_forecaster.device), batch_y.to(cnn_lstm_forecaster.device)
            optimizer.zero_grad()
            outputs = cnn_lstm_forecaster.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_sequences:.4f}")
        
    # Serialize model and scaler
    model_path = 'models/cnn_lstm_model.pt'
    torch.save(cnn_lstm_forecaster.model.state_dict(), model_path)
    joblib.dump(cnn_lstm_forecaster.scaler, model_path.replace('.pt', '_scaler.joblib'))
    print(f"CNNLSTM model saved to {model_path}")
    
    # Evaluate CNN-LSTM
    X_fore_test_scaled = cnn_lstm_forecaster.scaler.transform(X_fore_test)
    test_dataset = PyTorchSequenceDataset(X_fore_test_scaled, y_fore_test, seq_len=60)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    if len(test_dataset) > 0:
        cnn_lstm_forecaster.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(cnn_lstm_forecaster.device)
                preds_prob = cnn_lstm_forecaster.model(batch_x).cpu().numpy()
                all_preds.extend(preds_prob.flatten())
                all_targets.extend(batch_y.numpy().flatten())
                
        preds_bin = (np.array(all_preds) > 0.5).astype(int)
        y_seq_test = np.array(all_targets)
        
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_seq_test, preds_bin).ravel()
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
        tss_c = sens_c + spec_c - 1
        
        print("\nPyTorch CNN-LSTM Forecaster Classification Report:")
        print(classification_report(y_seq_test, preds_bin))
        print(f"Confusion Matrix: TN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}")
        print(f"TSS: {tss_c:.4f}, Sensitivity: {sens_c:.4f}, Specificity: {spec_c:.4f}")
        
    print("\n--------------------------------------------------")
    print("Training Pipeline Completed Successfully on Real 3-Year Data!")
    print("Model files generated in the models/ directory.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_training_pipeline()
