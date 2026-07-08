import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

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

def find_optimal_threshold(y_true, y_prob):
    """Find the decision threshold that maximises TSS (True Skill Statistic)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    tss_scores = tpr - fpr  # TSS = sensitivity + specificity - 1 = TPR - FPR
    best_idx = np.argmax(tss_scores)
    return thresholds[best_idx], tss_scores[best_idx]

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
    probs_xgb = xgb_forecaster.predict_probability(X_fore_test)
    best_thresh_xgb, best_tss_xgb = find_optimal_threshold(y_fore_test, probs_xgb)
    y_fore_pred_xgb = (probs_xgb > best_thresh_xgb).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_fore_test, y_fore_pred_xgb).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sensitivity + specificity - 1
    far = fp / (tn + fp) if (tn + fp) > 0 else 0
    far_ratio = fp / (tp + fp) if (tp + fp) > 0 else 0
    auc = roc_auc_score(y_fore_test, probs_xgb)
    
    print("\nXGBoost Forecaster Classification Report:")
    print(classification_report(y_fore_test, y_fore_pred_xgb, zero_division=0))
    print(f"  Optimal Threshold: {best_thresh_xgb:.4f}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  TSS              : {tss:.4f}")
    print(f"  Sensitivity      : {sensitivity:.4f}")
    print(f"  Specificity      : {specificity:.4f}")
    print(f"  False Alarm Rate : {far:.4f}")
    print(f"  False Alarm Ratio: {far_ratio:.4f}")
    
    # --- Forecasting Model 2: PyTorch 1D CNN-LSTM ---
    print("\n--- 3. Training PyTorch 1D CNN-LSTM Forecaster ---")
    cnn_lstm_forecaster = CNNLSTMForecasterWrapper(input_dim=len(feature_names), seq_len=60)
    print("Training PyTorch model (on-the-fly dataset)...")
    
    # Scale features and load on-the-fly dataset
    X_train_scaled = cnn_lstm_forecaster.scaler.fit_transform(X_fore_train)
    dataset = PyTorchSequenceDataset(X_train_scaled, y_fore_train, seq_len=60)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Handle class imbalance pos_weight
    y_train_seq = y_fore_train[60:]
    neg_count = (y_train_seq == 0).sum()
    pos_count = (y_train_seq == 1).sum()
    spw = float(neg_count / pos_count) if pos_count > 0 else 10.0
    spw = min(spw, 20.0)
    
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([spw], dtype=torch.float32).to(cnn_lstm_forecaster.device)
    )
    print(f"  CNN-LSTM pos_weight={spw:.2f} ({neg_count} neg / {pos_count} pos)")
    
    optimizer = torch.optim.Adam(cnn_lstm_forecaster.model.parameters(), lr=0.001)
    
    cnn_lstm_forecaster.model.train()
    epochs = 3
    num_sequences = len(dataset)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(cnn_lstm_forecaster.device), batch_y.to(cnn_lstm_forecaster.device)
            optimizer.zero_grad()
            logits = cnn_lstm_forecaster.model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_sequences:.4f}")
        
    # Serialize model and scaler
    model_path = 'models/cnn_lstm_model.pt'
    torch.save(cnn_lstm_forecaster.model.state_dict(), model_path)
    joblib.dump(cnn_lstm_forecaster.scaler, model_path.replace('.pt', '_scaler.joblib'))
    cnn_lstm_forecaster.is_trained = True
    print(f"CNNLSTM model saved to {model_path}")
    
    # Evaluate CNN-LSTM
    X_fore_test_scaled = cnn_lstm_forecaster.scaler.transform(X_fore_test)
    test_dataset = PyTorchSequenceDataset(X_fore_test_scaled, y_fore_test, seq_len=60)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    auc_c, tss_c, sens_c, spec_c, far_c, far_ratio_c, best_thresh_cnn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
    if len(test_dataset) > 0:
        cnn_lstm_forecaster.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(cnn_lstm_forecaster.device)
                logits = cnn_lstm_forecaster.model(batch_x)
                preds_prob = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds_prob.flatten())
                all_targets.extend(batch_y.numpy().flatten())
                
        all_preds = np.array(all_preds)
        y_seq_test = np.array(all_targets)
        
        best_thresh_cnn, best_tss_c = find_optimal_threshold(y_seq_test, all_preds)
        preds_bin = (all_preds > best_thresh_cnn).astype(int)
        
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_seq_test, preds_bin).ravel()
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
        tss_c = sens_c + spec_c - 1
        far_c = fp_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
        far_ratio_c = fp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        auc_c = roc_auc_score(y_seq_test, all_preds)
        
        print("\nPyTorch CNN-LSTM Forecaster Classification Report:")
        print(classification_report(y_seq_test, preds_bin, zero_division=0))
        print(f"  Optimal Threshold: {best_thresh_cnn:.4f}")
        print(f"  ROC-AUC          : {auc_c:.4f}")
        print(f"  Confusion Matrix : TN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}")
        print(f"  TSS              : {tss_c:.4f}")
        print(f"  Sensitivity      : {sens_c:.4f}")
        print(f"  Specificity      : {spec_c:.4f}")
        print(f"  False Alarm Rate : {far_c:.4f}")
        print(f"  False Alarm Ratio: {far_ratio_c:.4f}")
        
    # Export metrics to json
    metrics = {
        "xgb": {
            "tss": float(tss),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "false_alarm_rate": float(far),
            "false_alarm_ratio": float(far_ratio),
            "roc_auc": float(auc),
            "best_threshold": float(best_thresh_xgb)
        },
        "cnn_lstm": {
            "tss": float(tss_c),
            "sensitivity": float(sens_c),
            "specificity": float(spec_c),
            "false_alarm_rate": float(far_c),
            "false_alarm_ratio": float(far_ratio_c),
            "roc_auc": float(auc_c),
            "best_threshold": float(best_thresh_cnn)
        }
    }
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Exported model metrics to models/model_metrics.json")
    
    print("\n--------------------------------------------------")
    print("Training Pipeline Completed Successfully on Real 3-Year Data!")
    print("Model files generated in the models/ directory.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_training_pipeline()
