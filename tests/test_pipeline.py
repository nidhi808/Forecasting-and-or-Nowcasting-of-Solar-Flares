import os
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.noise_filter import apply_savgol_filter, clean_time_series
from src.data.cadence_sync import sync_cadences
from src.data.fits_processor import read_l1_fits
from src.data.mock_generator import generate_mock_data
from src.nowcasting.nowcaster import detect_flares_rule_based
from src.forecasting.forecaster import engineer_forecasting_features, create_labels

def test_savgol_filter():
    # Create time series with a random massive spike
    series = pd.Series([1.0, 1.1, 10.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.2])
    # The spike is at index 2
    filtered = apply_savgol_filter(series, window_length=5, polyorder=2)
    
    assert len(filtered) == len(series)
    # The spike at index 2 should be significantly smoothed
    assert filtered.iloc[2] < 6.0
    assert filtered.iloc[2] > 1.0

def test_cadence_sync():
    dates1 = pd.date_range("2026-07-03 00:00:00", periods=5, freq="1s")
    df1 = pd.DataFrame({"timestamp": dates1, "flux_solexs": [1.0, 1.1, np.nan, 1.3, 1.4]})
    
    # 2s cadence for HEL1OS
    dates2 = pd.date_range("2026-07-03 00:00:00", periods=3, freq="2s")
    df2 = pd.DataFrame({"timestamp": dates2, "flux_hel1os": [0.5, np.nan, 0.7]})
    
    synced = sync_cadences(df1, df2)
    
    assert len(synced) == 5
    assert 'flux_solexs' in synced.columns
    assert 'flux_hel1os' in synced.columns
    # NaNs should be interpolated or forward filled (since gap is less than max_gap=10s)
    assert not synced['flux_solexs'].isnull().any()
    assert not synced['flux_hel1os'].isnull().any()

def test_fits_processing_and_mock(tmp_path):
    # Generate mock FITS files in a temp directory
    solexs_fits, hel1os_fits = generate_mock_data(
        start_time_str="2026-07-03T00:00:00", 
        duration_seconds=120, 
        output_dir=str(tmp_path)
    )
    
    assert os.path.exists(solexs_fits)
    assert os.path.exists(hel1os_fits)
    
    # Read FITS files
    df_sxr, meta_sxr = read_l1_fits(solexs_fits)
    df_hxr, meta_hxr = read_l1_fits(hel1os_fits)
    
    assert not df_sxr.empty
    assert 'flux_solexs' in df_sxr.columns
    assert meta_sxr['instrument'] == 'SoLEXS'
    
    assert not df_hxr.empty
    assert 'flux_hel1os' in df_hxr.columns
    assert meta_hxr['instrument'] == 'HEL1OS'

def test_nowcasting():
    # Create simulated flare profile: background, sharp rise, slow decay
    timestamps = pd.date_range("2026-07-03 00:00:00", periods=100, freq="1s")
    
    flux_sxr = np.ones(100) * 1e-7
    # inject flare peaking at index 40 with peak flux 5e-5 (M-class)
    for i in range(100):
        if 20 <= i < 40:
            flux_sxr[i] = 1e-7 + (5e-5 - 1e-7) * np.exp((i - 40) / 5)
        elif i >= 40:
            flux_sxr[i] = 1e-7 + (5e-5 - 1e-7) * np.exp(-(i - 40) / 15)
            
    df = pd.DataFrame({
        "timestamp": timestamps,
        "flux_solexs_clean": flux_sxr,
        "flux_hel1os_clean": np.ones(100) * 1e-8
    })
    
    events = detect_flares_rule_based(df, min_duration_seconds=20)
    
    assert len(events) > 0
    # Peak flux should match our injected value approx
    assert events.iloc[0]['goes_class'] == 'M-Class'
    # Peak time should be near index 40
    expected_peak_time = timestamps[40]
    assert abs((events.iloc[0]['peak_time'] - expected_peak_time).total_seconds()) <= 2

def test_forecasting_helpers():
    timestamps = pd.date_range("2026-07-03 00:00:00", periods=100, freq="1s")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "flux_solexs_clean": np.ones(100) * 1e-7,
        "flux_hel1os_clean": np.ones(100) * 1e-8
    })
    
    df_feat, feature_names = engineer_forecasting_features(df)
    
    # 4 windows * 8 features + 4 baseline = 36 features
    assert len(feature_names) == 36
    assert df_feat.shape[0] == 100
    # Check that feature columns exist in df_feat
    for col in feature_names:
        assert col in df_feat.columns
