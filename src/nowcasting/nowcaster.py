import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def compute_derivatives(df, col='flux_solexs_clean', dt=1):
    """
    Computes first-order derivative (dx/dt) of the specified column.
    """
    flux = df[col].values
    # Compute derivative using forward/central differences
    deriv = np.zeros_like(flux)
    deriv[1:-1] = (flux[2:] - flux[:-2]) / (2 * dt)
    deriv[0] = (flux[1] - flux[0]) / dt
    deriv[-1] = (flux[-1] - flux[-2]) / dt
    return deriv

def detect_flares_rule_based(
    df,
    sxr_col='flux_solexs_clean',
    hxr_col='flux_hel1os_clean',
    dt=1,
    min_duration_seconds=60,
    min_peak_flux_sxr=1e-6,
    min_peak_flux_hxr=1e-7
):
    """
    Algorithmic Nowcasting: Detects flares independently in soft and hard X-rays.
    Uses first-derivative rate of change and identifies start, peak, and stop times.

    Args:
        min_duration_seconds: Minimum flare duration to register (default 60s).
            Filters out cosmic ray spikes and sub-second noise bursts.
        min_peak_flux_sxr: Minimum SXR peak flux to register (default 1e-6 = C-class).
        min_peak_flux_hxr: Minimum HXR peak flux to register (default 1e-7).
    """
    events = []
    # Ensure integer index for safe .loc[] access
    df = df.reset_index(drop=True)

    # Calculate derivatives
    deriv_sxr = compute_derivatives(df, sxr_col, dt)
    deriv_hxr = compute_derivatives(df, hxr_col, dt)
    
    # Thresholds for rate of change (tuned based on background levels)
    # Background is typically ~1e-7, so a rise of > 2e-8 per second is a good indicator
    sxr_threshold = 1.5e-8
    hxr_threshold = 3.0e-9
    
    # Detection loop for SXR (SoLEXS)
    in_flare_sxr = False
    start_idx_sxr = None
    peak_idx_sxr = None
    peak_flux_sxr = 0.0
    
    for i in range(1, len(df)):
        flux = df.loc[i, sxr_col]
        if pd.isna(flux):
            continue
            
        d = deriv_sxr[i]
        
        if not in_flare_sxr:
            # Flare starts when derivative crosses threshold
            if d > sxr_threshold:
                in_flare_sxr = True
                start_idx_sxr = i
                peak_idx_sxr = i
                peak_flux_sxr = flux
        else:
            # Update peak if current flux is higher
            if flux > peak_flux_sxr:
                peak_flux_sxr = flux
                peak_idx_sxr = i
                
            # Flare ends when derivative returns to near zero and flux decays
            # Or if it starts rising again (new flare)
            if d < 1e-9 and flux < (peak_flux_sxr * 0.5 + df.loc[start_idx_sxr, sxr_col] * 0.5):
                duration_s = (df.loc[i, 'timestamp'] - df.loc[start_idx_sxr, 'timestamp']).total_seconds()

                # Quality gate: reject noise spikes (cosmic rays, sub-minute bursts)
                if peak_flux_sxr >= min_peak_flux_sxr and duration_s >= min_duration_seconds:
                    # Classify flare using GOES scale
                    if peak_flux_sxr >= 1e-4:
                        goes_class = "X-Class"
                    elif peak_flux_sxr >= 1e-5:
                        goes_class = "M-Class"
                    elif peak_flux_sxr >= 1e-6:
                        goes_class = "C-Class"
                    else:
                        goes_class = "B-Class / Sub-flare"

                    events.append({
                        "instrument": "SoLEXS (Soft X-ray)",
                        "start_time": df.loc[start_idx_sxr, 'timestamp'],
                        "peak_time": df.loc[peak_idx_sxr, 'timestamp'],
                        "end_time": df.loc[i, 'timestamp'],
                        "peak_flux": peak_flux_sxr,
                        "goes_class": goes_class,
                        "duration_seconds": duration_s
                    })
                in_flare_sxr = False
                peak_flux_sxr = 0.0
                
    # Detection loop for HXR (HEL1OS)
    in_flare_hxr = False
    start_idx_hxr = None
    peak_idx_hxr = None
    peak_flux_hxr = 0.0
    
    for i in range(1, len(df)):
        flux = df.loc[i, hxr_col]
        if pd.isna(flux):
            continue
            
        d = deriv_hxr[i]
        
        if not in_flare_hxr:
            if d > hxr_threshold:
                in_flare_hxr = True
                start_idx_hxr = i
                peak_idx_hxr = i
                peak_flux_hxr = flux
        else:
            if flux > peak_flux_hxr:
                peak_flux_hxr = flux
                peak_idx_hxr = i
                
            if d < 5e-10 and flux < (peak_flux_hxr * 0.3 + df.loc[start_idx_hxr, hxr_col] * 0.7):
                duration_s = (df.loc[i, 'timestamp'] - df.loc[start_idx_hxr, 'timestamp']).total_seconds()

                # Quality gate: reject noise spikes
                if peak_flux_hxr >= min_peak_flux_hxr and duration_s >= min_duration_seconds:
                    if peak_flux_hxr >= 1e-5:
                        goes_class = "X-Class Equivalent"
                    elif peak_flux_hxr >= 1e-6:
                        goes_class = "M-Class Equivalent"
                    elif peak_flux_hxr >= 1e-7:
                        goes_class = "C-Class Equivalent"
                    else:
                        goes_class = "Sub-flare"

                    events.append({
                        "instrument": "HEL1OS (Hard X-ray)",
                        "start_time": df.loc[start_idx_hxr, 'timestamp'],
                        "peak_time": df.loc[peak_idx_hxr, 'timestamp'],
                        "end_time": df.loc[i, 'timestamp'],
                        "peak_flux": peak_flux_hxr,
                        "goes_class": goes_class,
                        "duration_seconds": duration_s
                    })
                in_flare_hxr = False
                peak_flux_hxr = 0.0
                
    # Merge independent catalogues to generate a master catalogue
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values('peak_time').reset_index(drop=True)
        
    return events_df

class MLNowcaster:
    """
    ML-based Nowcasting: Uses derivatives and current flux levels to classify 
    the active state of the Sun (0: Quiet, 1: C-Class, 2: M-Class, 3: X-Class).
    Uses a Random Forest or XGBoost Classifier trained on derivative features.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
        self.is_trained = False
        
    def extract_features(self, df, sxr_col='flux_solexs_clean', hxr_col='flux_hel1os_clean'):
        df = df.copy()
        df['deriv_sxr'] = compute_derivatives(df, sxr_col)
        df['deriv_hxr'] = compute_derivatives(df, hxr_col)
        df['deriv2_sxr'] = np.gradient(df['deriv_sxr'])
        df['deriv2_hxr'] = np.gradient(df['deriv_hxr'])
        
        # Features: current fluxes, derivatives, and second derivatives
        feature_cols = [sxr_col, hxr_col, 'deriv_sxr', 'deriv_hxr', 'deriv2_sxr', 'deriv2_hxr']
        X = df[feature_cols].ffill().fillna(0.0).values
        return X
        
    def train(self, df, labels, model_path='models/nowcaster_model.joblib'):
        """
        Trains the nowcaster using the derivative features.
        Labels: 0 (Quiet), 1 (C), 2 (M), 3 (X).
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        X = self.extract_features(df)
        self.model.fit(X, labels)
        self.is_trained = True
        joblib.dump(self.model, model_path)
        print(f"Nowcasting model saved to {model_path}")
        
    def load(self, model_path='models/nowcaster_model.joblib'):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
            print(f"Nowcasting model loaded from {model_path}")
        else:
            print("Model path does not exist. Using untrained model.")
            
    def predict(self, df):
        """
        Predicts solar flare class for the latest timestamps in the dataframe.
        """
        X = self.extract_features(df)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        class_map = {0: 'Quiet', 1: 'C-Class', 2: 'M-Class', 3: 'X-Class'}
        pred_labels = [class_map[p] for p in preds]
        
        return pred_labels, probs
