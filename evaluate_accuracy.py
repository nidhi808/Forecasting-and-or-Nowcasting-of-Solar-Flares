"""
Standalone accuracy evaluation for the Solar Flare pipeline.
Trains all three models on 2-day mock data and prints full metrics.
CNN-LSTM is limited to 3 epochs for speed.
"""
import sys, io
# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data.mock_generator import generate_flare_profile
from src.data.cadence_sync import sync_cadences
from src.data.noise_filter import apply_savgol_filter
from src.nowcasting.nowcaster import detect_flares_rule_based, MLNowcaster
from src.forecasting.forecaster import (
    engineer_forecasting_features, create_labels,
    XGBForecaster, CNNLSTMForecasterWrapper, prepare_sequence_data
)

SEP  = "=" * 60
SEP2 = "-" * 60

def log(msg):
    print(msg, flush=True)

def create_nowcast_labels(df, events):
    y = np.zeros(len(df), dtype=int)
    if events is None or events.empty:
        return y
    for _, flare in events.sort_values('peak_flux').iterrows():
        gc = str(flare['goes_class'])
        lbl = 3 if 'X-Class' in gc else (2 if 'M-Class' in gc else (1 if 'C-Class' in gc else 0))
        if lbl > 0:
            mask = (df['timestamp'] >= flare['start_time']) & (df['timestamp'] <= flare['end_time'])
            y[mask] = lbl
    return y

def print_confusion(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1          = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    tss         = sensitivity + specificity - 1
    log(f"  Confusion Matrix : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    log(f"  Sensitivity      : {sensitivity:.4f}   (Recall / True Positive Rate)")
    log(f"  Specificity      : {specificity:.4f}   (True Negative Rate)")
    log(f"  Precision        : {precision:.4f}")
    log(f"  F1 Score         : {f1:.4f}")
    log(f"  TSS              : {tss:.4f}   (True Skill Statistic, key metric for flare forecasting)")

# ── 1. Generate mock 2-day dataset ─────────────────────────────────────────
log("=" * 60)
log("  Solar Flare Model Accuracy Evaluation")
log("=" * 60)
log("\n[1/5] Generating 2-day mock telemetry data...")

duration_seconds = 86400 * 2
start_time = datetime.fromisoformat("2026-07-01T00:00:00")
t_sxr = np.arange(0, duration_seconds, 1)
t_hxr = np.arange(0, duration_seconds, 2)

np.random.seed(0)
bg_sxr = 1.2e-7 + np.random.normal(0, 1e-9, len(t_sxr)) + 2e-8 * np.sin(2*np.pi*t_sxr/43200)
bg_hxr = 2.0e-8 + np.random.normal(0, 5e-10, len(t_hxr)) + 3e-9 * np.sin(2*np.pi*t_hxr/43200)

flares_schedule = [
    {"peak_t": 15000, "peak_flux_sxr": 4e-6,  "peak_flux_hxr": 8e-7,  "rise": 400, "decay": 1500},
    {"peak_t": 50000, "peak_flux_sxr": 3e-5,  "peak_flux_hxr": 8e-6,  "rise": 600, "decay": 2500},
    {"peak_t": 95000, "peak_flux_sxr": 1.5e-4,"peak_flux_hxr": 5e-5,  "rise": 800, "decay": 4000},
    {"peak_t": 130000,"peak_flux_sxr": 6e-6,  "peak_flux_hxr": 1.2e-6,"rise": 350, "decay": 1200},
    {"peak_t": 155000,"peak_flux_sxr": 2e-5,  "peak_flux_hxr": 5e-6,  "rise": 500, "decay": 2000},
]

sxr_flare_flux = np.zeros_like(t_sxr, dtype=float)
hxr_flare_flux = np.zeros_like(t_hxr, dtype=float)
for f in flares_schedule:
    sxr_flare_flux += generate_flare_profile(t_sxr, f["peak_t"], f["peak_flux_sxr"], f["rise"], f["decay"])
    hxr_flare_flux += generate_flare_profile(t_hxr, f["peak_t"]-60, f["peak_flux_hxr"], f["rise"]//2, f["decay"]//5)

flux_sxr = bg_sxr + sxr_flare_flux
flux_hxr = bg_hxr + hxr_flare_flux

# inject cosmic ray spikes
flux_sxr[np.random.choice(len(t_sxr), 800, replace=False)] += np.random.uniform(5e-6, 5e-5, 800)
flux_hxr[np.random.choice(len(t_hxr), 500, replace=False)] += np.random.uniform(1e-6, 1.5e-5, 500)

times_sxr = [start_time + timedelta(seconds=int(s)) for s in t_sxr]
times_hxr = [start_time + timedelta(seconds=int(s)) for s in t_hxr]

df_sxr = pd.DataFrame({"timestamp": times_sxr, "flux_solexs": flux_sxr})
df_hxr = pd.DataFrame({"timestamp": times_hxr, "flux_hel1os": flux_hxr})
log(f"  SXR rows: {len(df_sxr):,}   HXR rows: {len(df_hxr):,}")

# ── 2. Sync & clean ─────────────────────────────────────────────────────────
log("\n[2/5] Syncing cadences & cleaning noise...")
df_sync = sync_cadences(df_sxr, df_hxr)
df_sync['flux_solexs_clean'] = apply_savgol_filter(df_sync['flux_solexs'], window_length=31)
df_sync['flux_hel1os_clean'] = apply_savgol_filter(df_sync['flux_hel1os'], window_length=15)
log(f"  Synced rows: {len(df_sync):,}")

# ── 3. Event detection & labels ──────────────────────────────────────────────
log("\n[3/5] Rule-based nowcasting & label generation...")
events_raw = detect_flares_rule_based(df_sync)

# Filter out noise-triggered micro-events:
#   - minimum duration 60s (sub-minute bursts are likely cosmic rays)
#   - minimum peak flux: SXR >= 1e-6 (C-class threshold), HXR >= 1e-7
if not events_raw.empty:
    sxr_mask = (events_raw['instrument'].str.contains('SoLEXS')) & \
               (events_raw['peak_flux'] >= 1e-6) & \
               (events_raw['duration_seconds'] >= 60)
    hxr_mask = (events_raw['instrument'].str.contains('HEL1OS')) & \
               (events_raw['peak_flux'] >= 1e-7) & \
               (events_raw['duration_seconds'] >= 60)
    events = events_raw[sxr_mask | hxr_mask].reset_index(drop=True)
else:
    events = events_raw

log(f"  Raw events detected  : {len(events_raw)}")
log(f"  After quality filter : {len(events)} (duration>=60s, flux>=C-class)")
if not events.empty:
    log(events[['instrument','peak_time','goes_class','peak_flux','duration_seconds']].to_string(index=False))

nowcast_labels    = create_nowcast_labels(df_sync, events)
forecasting_labels = create_labels(df_sync, events)
pos_ratio = forecasting_labels.sum() / len(forecasting_labels) * 100
log(f"\n  Forecasting label distribution: {forecasting_labels.sum():,} positive / {len(forecasting_labels):,} total ({pos_ratio:.2f}%)")

# ── 4. Feature engineering & split ──────────────────────────────────────────
log("\n[4/5] Feature engineering...")
df_feat, feature_names = engineer_forecasting_features(df_sync)
log(f"  Features: {len(feature_names)}")

split_idx = int(0.7 * len(df_sync))
log(f"  Train rows: {split_idx:,}  |  Test rows: {len(df_sync)-split_idx:,}")

# ── 5. MODEL EVALUATIONS ─────────────────────────────────────────────────────
log("\n[5/5] Training & Evaluating Models...")

# ── Model A: Random Forest MLNowcaster ──────────────────────────────────────
log("\n" + SEP)
log("  MODEL A: Random Forest MLNowcaster (4-class)")
log(SEP2)
nowcaster = MLNowcaster()
X_now = nowcaster.extract_features(df_sync)
X_now_train, X_now_test = X_now[:split_idx], X_now[split_idx:]
y_now_train, y_now_test = nowcast_labels[:split_idx], nowcast_labels[split_idx:]

nowcaster.train(df_sync.iloc[:split_idx].reset_index(drop=True), y_now_train,
                model_path='models/nowcaster_model.joblib')

y_now_pred = nowcaster.model.predict(X_now_test)
unique_labels = np.unique(y_now_test)
class_names = ['Quiet', 'C-Class', 'M-Class', 'X-Class']
target_names = [class_names[i] for i in unique_labels]

log("\n  Classification Report:")
log(classification_report(y_now_test, y_now_pred, target_names=target_names, zero_division=0))
overall_acc = (y_now_pred == y_now_test).mean()
log(f"  Overall Accuracy: {overall_acc*100:.2f}%")

# ── Model B: XGBoost Forecaster ─────────────────────────────────────────────
log("\n" + SEP)
log("  MODEL B: XGBoost Forecaster (binary: flare in 15-30m?)")
log(SEP2)
X_fore = df_feat[feature_names].values
y_fore = forecasting_labels

X_fore_train, X_fore_test = X_fore[:split_idx], X_fore[split_idx:]
y_fore_train, y_fore_test = y_fore[:split_idx], y_fore[split_idx:]

xgb_fc = XGBForecaster()
neg_c = (y_fore_train == 0).sum()
pos_c = (y_fore_train == 1).sum()
xgb_fc.model.scale_pos_weight = neg_c / pos_c if pos_c > 0 else 10.0
xgb_fc.train(X_fore_train, y_fore_train, model_path='models/xgboost_forecaster.json')

probs_xgb = xgb_fc.predict_probability(X_fore_test)
preds_xgb = (probs_xgb > 0.5).astype(int)

log("\n  Classification Report:")
log(classification_report(y_fore_test, preds_xgb, zero_division=0))

if len(np.unique(y_fore_test)) > 1:
    auc = roc_auc_score(y_fore_test, probs_xgb)
    log(f"  ROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_fore_test, preds_xgb)
if cm.shape == (2,2):
    tn, fp, fn, tp = cm.ravel()
    print_confusion(tn, fp, fn, tp)

# ── Model C: CNN-LSTM (3 epochs for speed) ──────────────────────────────────
log("\n" + SEP)
log("  MODEL C: PyTorch 1D CNN-LSTM Forecaster (3 epochs)")
log(SEP2)
cnn_lstm = CNNLSTMForecasterWrapper(input_dim=len(feature_names), seq_len=60)
cnn_lstm.train(X_fore_train, y_fore_train, epochs=3,
               batch_size=256, model_path='models/cnn_lstm_model.pt')

X_test_scaled = cnn_lstm.scaler.transform(X_fore_test)
X_seq_test, y_seq_test = prepare_sequence_data(X_test_scaled, y_fore_test, seq_len=60)

if len(X_seq_test) > 0:
    cnn_lstm.model.eval()
    with torch.no_grad():
        seq_t = torch.tensor(X_seq_test, dtype=torch.float32).to(cnn_lstm.device)
        probs_cnn = cnn_lstm.model(seq_t).cpu().numpy()
    preds_cnn = (probs_cnn > 0.5).astype(int)

    log("\n  Classification Report:")
    log(classification_report(y_seq_test, preds_cnn, zero_division=0))

    if len(np.unique(y_seq_test)) > 1:
        auc_c = roc_auc_score(y_seq_test, probs_cnn)
        log(f"  ROC-AUC: {auc_c:.4f}")

    cm_c = confusion_matrix(y_seq_test, preds_cnn)
    if cm_c.shape == (2,2):
        tn_c, fp_c, fn_c, tp_c = cm_c.ravel()
        print_confusion(tn_c, fp_c, fn_c, tp_c)

log("\n" + "=" * 60)
log("  Evaluation Complete. Models saved to models/")
log("=" * 60)
