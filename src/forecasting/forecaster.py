import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Feature engineering helper
def engineer_forecasting_features(df, sxr_col='flux_solexs_clean', hxr_col='flux_hel1os_clean', cadence_seconds=1):
    """
    Engineers temporal rolling features for solar flare forecasting.
    Includes rolling averages, maximum derivatives, standard deviations, and rate of change
    over multiple historical windows (5 mins, 10 mins, 30 mins, 60 mins).
    """
    df = df.copy()
    
    # Calculate derivatives first
    df['deriv_sxr'] = np.gradient(df[sxr_col].ffill().fillna(0.0))
    df['deriv_hxr'] = np.gradient(df[hxr_col].ffill().fillna(0.0))
    
    windows = {
        '5m': int(300 / cadence_seconds),
        '10m': int(600 / cadence_seconds),
        '30m': int(1800 / cadence_seconds),
        '60m': int(3600 / cadence_seconds)
    }
    
    feature_cols = []
    
    # Generate rolling features
    for name, size in windows.items():
        if size > len(df):
            size = max(2, len(df) // 2)
            
        # SoLEXS rolling features
        df[f'sxr_mean_{name}'] = df[sxr_col].rolling(window=size, min_periods=1).mean()
        df[f'sxr_std_{name}'] = df[sxr_col].rolling(window=size, min_periods=1).std().fillna(0)
        df[f'sxr_max_{name}'] = df[sxr_col].rolling(window=size, min_periods=1).max()
        df[f'sxr_max_deriv_{name}'] = df['deriv_sxr'].rolling(window=size, min_periods=1).max()
        
        # HEL1OS rolling features
        df[f'hxr_mean_{name}'] = df[hxr_col].rolling(window=size, min_periods=1).mean()
        df[f'hxr_std_{name}'] = df[hxr_col].rolling(window=size, min_periods=1).std().fillna(0)
        df[f'hxr_max_{name}'] = df[hxr_col].rolling(window=size, min_periods=1).max()
        df[f'hxr_max_deriv_{name}'] = df['deriv_hxr'].rolling(window=size, min_periods=1).max()
        
        feature_cols.extend([
            f'sxr_mean_{name}', f'sxr_std_{name}', f'sxr_max_{name}', f'sxr_max_deriv_{name}',
            f'hxr_mean_{name}', f'hxr_std_{name}', f'hxr_max_{name}', f'hxr_max_deriv_{name}'
        ])
        
    # Baseline features
    feature_cols.extend([sxr_col, hxr_col, 'deriv_sxr', 'deriv_hxr'])
    
    return df, feature_cols

def create_labels(df, nowcast_events, forecast_window_minutes=30, lead_time_minutes=15):
    """
    Creates binary labels for training a forecasting model.
    A timestamp t is labeled 1 (flare imminent) if a flare peak occurs in the interval
    [t + lead_time, t + lead_time + forecast_window], and 0 otherwise.
    """
    y = np.zeros(len(df), dtype=int)
    
    if nowcast_events is None or nowcast_events.empty:
        return y
        
    # Filter to actual significant flares (C-class and above)
    flares = nowcast_events[nowcast_events['goes_class'].isin(['C-Class', 'M-Class', 'X-Class'])]
    
    for _, flare in flares.iterrows():
        peak_time = flare['peak_time']
        
        # Determine the trigger window prior to the peak
        trigger_start = peak_time - pd.Timedelta(minutes=forecast_window_minutes + lead_time_minutes)
        trigger_end = peak_time - pd.Timedelta(minutes=lead_time_minutes)
        
        # Mark all timestamps within the trigger window as positive
        mask = (df['timestamp'] >= trigger_start) & (df['timestamp'] <= trigger_end)
        y[mask] = 1
        
    return y


# XGBoost Forecasting Model
class XGBForecaster:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=10, # handles class imbalance (98% quiet time)
            eval_metric='logloss',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X, y, model_path='models/xgboost_forecaster.json'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model and scaler
        self.model.save_model(model_path)
        import joblib
        joblib.dump(self.scaler, model_path.replace('.json', '_scaler.joblib'))
        print(f"XGBoost model saved to {model_path}")
        
    def load(self, model_path='models/xgboost_forecaster.json'):
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            import joblib
            scaler_path = model_path.replace('.json', '_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"XGBoost model loaded from {model_path}")
        else:
            print("XGBoost model path not found. Model is untrained.")
            
    def predict_probability(self, X):
        if not self.is_trained:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        # Returns probability of class 1 (flare imminent)
        return self.model.predict_proba(X_scaled)[:, 1]


# PyTorch 1D CNN-LSTM Hybrid Model
class CNNLSTMForecaster(nn.Module):
    def __init__(self, input_dim=10, cnn_out_channels=16, lstm_hidden_dim=32, seq_len=60):
        super(CNNLSTMForecaster, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 1D CNN to extract shape/gradient features across the window
        # Input shape: (batch, input_dim, seq_len)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Reduced sequence length after MaxPool1d
        reduced_seq_len = seq_len // 4
        
        # LSTM to model temporal sequence momentum
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels*2,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected predictor
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        # Transpose for PyTorch Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN forward
        cnn_out = self.cnn(x) # (batch, cnn_out_channels*2, reduced_seq_len)
        
        # Transpose back for LSTM: (batch, reduced_seq_len, features)
        lstm_in = cnn_out.transpose(1, 2)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_in) # (batch, reduced_seq_len, lstm_hidden_dim)
        
        # Use final LSTM time step output
        out = self.fc(lstm_out[:, -1, :]) # (batch, 1)
        return out.squeeze(-1)

# Helper function to create sequence datasets for CNN-LSTM
def prepare_sequence_data(X, y, seq_len=60):
    """
    Prepares sliding window sequences for PyTorch models.
    X: shape (num_samples, num_features)
    y: shape (num_samples,)
    """
    X_seqs = []
    y_seqs = []
    for i in range(len(X) - seq_len):
        X_seqs.append(X[i : i + seq_len])
        y_seqs.append(y[i + seq_len])
        
    return np.array(X_seqs), np.array(y_seqs)


# Wrapper for CNN-LSTM
class CNNLSTMForecasterWrapper:
    def __init__(self, input_dim=36, seq_len=60):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMForecaster(input_dim=input_dim, seq_len=seq_len).to(self.device)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X, y, epochs=10, batch_size=128, lr=0.001, model_path='models/cnn_lstm_model.pt'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sliding windows
        X_seqs, y_seqs = prepare_sequence_data(X_scaled, y, self.seq_len)
        
        if len(X_seqs) == 0:
            print("Not enough sequence data to train CNN-LSTM.")
            return
            
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        train_ds = TensorDataset(torch.tensor(X_seqs, dtype=torch.float32), torch.tensor(y_seqs, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # scale_pos_weight calculation for loss balancing
        neg_count = sum(y == 0)
        pos_count = sum(y == 1)
        spw = neg_count / pos_count if pos_count > 0 else 10.0
        weight_tensor = torch.tensor([float(spw)]).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                
                # Weighted BCELoss because Sigmoid is in the forward function
                loss = -torch.mean(weight_tensor * batch_y * torch.log(preds + 1e-7) + (1 - batch_y) * torch.log(1 - preds + 1e-7))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(X_seqs):.4f}")
            
        self.is_trained = True
        
        # Save model and scaler
        torch.save(self.model.state_dict(), model_path)
        import joblib
        joblib.dump(self.scaler, model_path.replace('.pt', '_scaler.joblib'))
        print(f"CNNLSTM model saved to {model_path}")
        
    def load(self, model_path='models/cnn_lstm_model.pt'):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            import joblib
            scaler_path = model_path.replace('.pt', '_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"CNNLSTM model loaded from {model_path}")
        else:
            print("CNNLSTM model path not found. Model is untrained.")
            
    def predict_probability(self, X):
        """
        Expects X of shape (seq_len, num_features) or (N, num_features) representing the history.
        """
        if not self.is_trained:
            return 0.0
            
        if len(X) < self.seq_len:
            return 0.0
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Take the latest seq_len steps
        seq = X_scaled[-self.seq_len:]
        
        # Format for model: shape (1, seq_len, num_features)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prob = self.model(seq_tensor).cpu().numpy()[0]
        return float(prob)
