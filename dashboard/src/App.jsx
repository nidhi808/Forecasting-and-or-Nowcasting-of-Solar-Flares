import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  ShieldAlert, 
  UploadCloud, 
  FileText, 
  Compass, 
  Clock, 
  RefreshCw, 
  Layers 
} from 'lucide-react';
import Plotly from 'plotly.js-dist-min';

function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [catalog, setCatalog] = useState([]);
  const [forecast, setForecast] = useState({ probability: 0.05, status: "Quiet", nowcast_class: "Quiet" });
  const [modelType, setModelType] = useState('xgb');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Upload state
  const [solexsFile, setSolexsFile] = useState(null);
  const [hel1osFile, setHel1osFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const chartRef = useRef(null);
  const uploadChartRef = useRef(null);

  const backendUrl = "http://127.0.0.1:8000";

  // Fetch telemetry, forecast and catalog from backend API
  const fetchData = async () => {
    try {
      // Fetch live telemetry
      const telRes = await fetch(`${backendUrl}/api/telemetry/live?limit=1500`);
      if (!telRes.ok) throw new Error("Failed to fetch live telemetry");
      const telData = await telRes.json();
      setTelemetry(telData);

      // Fetch catalog
      const catRes = await fetch(`${backendUrl}/api/nowcast/catalog`);
      if (!catRes.ok) throw new Error("Failed to fetch nowcast catalog");
      const catData = await catRes.json();
      setCatalog(catData);

      // Fetch forecast probability
      const foreRes = await fetch(`${backendUrl}/api/predict?model_type=${modelType}`);
      if (!foreRes.ok) throw new Error("Failed to fetch forecast prediction");
      const foreData = await foreRes.json();
      setForecast(foreData);

      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to connect to telemetry server. Please make sure the FastAPI server is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Poll every 5 seconds for real-time telemetry updates
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [modelType]);

  // Update telemetry Plotly chart
  useEffect(() => {
    if (!chartRef.current || telemetry.length === 0) return;

    const timestamps = telemetry.map(t => t.timestamp);
    
    // Trace 1: SoLEXS raw count rate
    const traceSolexsRaw = {
      x: timestamps,
      y: telemetry.map(t => t.flux_solexs),
      name: 'SoLEXS Raw (SXR)',
      type: 'scattergl',
      mode: 'lines',
      line: { color: 'rgba(58, 134, 200, 0.35)', width: 1 },
      yaxis: 'y'
    };

    // Trace 2: SoLEXS smoothed (Savitzky-Golay)
    const traceSolexsClean = {
      x: timestamps,
      y: telemetry.map(t => t.flux_solexs_clean),
      name: 'SoLEXS Clean (SXR)',
      type: 'scattergl',
      mode: 'lines',
      line: { color: '#ff9f1c', width: 2 },
      yaxis: 'y'
    };

    // Trace 3: HEL1OS raw count rate
    const traceHel1osRaw = {
      x: timestamps,
      y: telemetry.map(t => t.flux_hel1os),
      name: 'HEL1OS Raw (HXR)',
      type: 'scattergl',
      mode: 'lines',
      line: { color: 'rgba(255, 77, 77, 0.3)', width: 1 },
      yaxis: 'y2'
    };

    // Trace 4: HEL1OS smoothed (Savitzky-Golay)
    const traceHel1osClean = {
      x: timestamps,
      y: telemetry.map(t => t.flux_hel1os_clean),
      name: 'HEL1OS Clean (HXR)',
      type: 'scattergl',
      mode: 'lines',
      line: { color: '#ff4d4d', width: 2 },
      yaxis: 'y2'
    };

    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 60, r: 60, t: 30, b: 40 },
      showlegend: true,
      legend: { orientation: 'h', y: 1.12, font: { color: '#8a8fa3' } },
      xaxis: {
        gridcolor: 'rgba(255,255,255,0.05)',
        tickcolor: 'rgba(255,255,255,0.2)',
        tickfont: { color: '#8a8fa3' },
        type: 'date'
      },
      yaxis: {
        title: { text: 'Soft X-Ray Flux (W/m²)', font: { color: '#ff9f1c' } },
        gridcolor: 'rgba(255,255,255,0.05)',
        tickfont: { color: '#ff9f1c' },
        exponentformat: 'e',
        showexponent: 'all'
      },
      yaxis2: {
        title: { text: 'Hard X-Ray Flux (W/m²)', font: { color: '#ff4d4d' } },
        tickfont: { color: '#ff4d4d' },
        overlaying: 'y',
        side: 'right',
        showgrid: false,
        exponentformat: 'e',
        showexponent: 'all'
      }
    };

    const config = { responsive: true, displaylogo: false };

    Plotly.newPlot(chartRef.current, [traceSolexsRaw, traceSolexsClean, traceHel1osRaw, traceHel1osClean], layout, config);
  }, [telemetry]);

  // Handle file uploads
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!solexsFile || !hel1osFile) {
      setUploadError("Please select both SoLEXS and HEL1OS Level-1 FITS files.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadResult(null);

    const formData = new FormData();
    formData.append("solexs", solexsFile);
    formData.append("hel1os", hel1osFile);

    try {
      const res = await fetch(`${backendUrl}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "FITS file processing failed");
      }

      const data = await res.json();
      setUploadResult(data);
      
      // Also refresh telemetry and catalog
      fetchData();
    } catch (err) {
      console.error(err);
      setUploadError(err.message || "Failed to process files");
    } finally {
      setUploading(false);
    }
  };

  // Status variables
  const currentSXR = telemetry.length > 0 ? telemetry[telemetry.length - 1].flux_solexs_clean : 0;
  const currentHXR = telemetry.length > 0 ? telemetry[telemetry.length - 1].flux_hel1os_clean : 0;
  
  // Flag active flare states
  const activeAlert = forecast.status !== "Quiet";
  const criticalAlert = forecast.status === "CRITICAL WARNING";

  return (
    <div className="min-h-screen text-slate-100 flex flex-col">
      {/* Header Cockpit */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="bg-amber-500/10 p-2 rounded-lg border border-amber-500/20">
            <Compass className="h-6 w-6 text-amber-500 rotate-sun" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight tracking-wider text-slate-50 uppercase">
              Aditya-L1 Heliophysics Dashboard
            </h1>
            <p className="text-xs text-slate-500">
              Joint Solar Low & High Energy X-ray Spectrometer Telemetry (SoLEXS & HEL1OS)
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {error ? (
            <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-500 text-xs">
              <span className="h-2 w-2 rounded-full bg-red-500 animate-ping" />
              <span>OFFLINE</span>
            </div>
          ) : (
            <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span>LIVE TELEMETRY ACTIVE</span>
            </div>
          )}
          
          <button 
            onClick={fetchData} 
            className="p-1.5 hover:bg-slate-800 rounded-lg border border-slate-800 text-slate-400 hover:text-slate-200 transition"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-6 space-y-6 max-w-[1600px] w-full mx-auto">
        
        {error && (
          <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 flex items-center space-x-3 text-sm">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Dashboard Status Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          
          {/* SXR Flux */}
          <div className="glass-panel p-5 flex flex-col justify-between glow-blue">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">SoLEXS (Soft X-Ray)</span>
              <span className="text-amber-500 bg-amber-500/10 p-1.5 rounded-md border border-amber-500/20">
                <Activity className="h-4 w-4" />
              </span>
            </div>
            <div className="my-4">
              <div className="text-2xl font-bold font-mono tracking-tight text-amber-500">
                {currentSXR ? currentSXR.toExponential(4) : "0.0000e-0"}
              </div>
              <div className="text-xs text-slate-500 mt-1">W/m² · 1-8 Å range</div>
            </div>
            <div className="text-xs text-slate-400 border-t border-slate-900 pt-2 flex items-center space-x-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
              <span>Thermal plasma heating</span>
            </div>
          </div>

          {/* HXR Flux */}
          <div className="glass-panel p-5 flex flex-col justify-between glow-orange">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">HEL1OS (Hard X-Ray)</span>
              <span className="text-red-500 bg-red-500/10 p-1.5 rounded-md border border-red-500/20">
                <Activity className="h-4 w-4" />
              </span>
            </div>
            <div className="my-4">
              <div className="text-2xl font-bold font-mono tracking-tight text-red-500">
                {currentHXR ? currentHXR.toExponential(4) : "0.0000e-0"}
              </div>
              <div className="text-xs text-slate-500 mt-1">W/m² · 10-25 keV range</div>
            </div>
            <div className="text-xs text-slate-400 border-t border-slate-900 pt-2 flex items-center space-x-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              <span>Non-thermal particle accel.</span>
            </div>
          </div>

          {/* Predictive Warning Panel */}
          <div className={`glass-panel p-5 flex flex-col justify-between ${criticalAlert ? 'pulse-alarm bg-red-950/20' : (activeAlert ? 'border-amber-500/40 bg-amber-950/10' : '')}`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Predictive Forecaster</span>
              <div className="flex items-center space-x-2">
                <select 
                  value={modelType} 
                  onChange={(e) => setModelType(e.target.value)}
                  className="text-[10px] bg-slate-905 border border-slate-800 rounded px-1.5 py-0.5 text-slate-300 focus:outline-none"
                >
                  <option value="xgb">XGBoost model</option>
                  <option value="cnn_lstm">CNN-LSTM (PyTorch)</option>
                </select>
                <span className={`${criticalAlert ? 'text-red-500 bg-red-500/10' : (activeAlert ? 'text-amber-500 bg-amber-500/10' : 'text-slate-500 bg-slate-500/10')} p-1.5 rounded-md border ${criticalAlert ? 'border-red-500/20' : (activeAlert ? 'border-amber-500/20' : 'border-slate-500/10')}`}>
                  <ShieldAlert className="h-4 w-4" />
                </span>
              </div>
            </div>
            <div className="my-4">
              <div className="flex items-baseline space-x-1">
                <span className="text-3xl font-extrabold font-mono tracking-tight">
                  {Math.round(forecast.probability * 100)}
                </span>
                <span className="text-lg font-bold text-slate-400">%</span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Flare Probability (15-30m Lead Time)
              </div>
            </div>
            <div className="text-xs border-t border-slate-900 pt-2 flex items-center justify-between">
              <span className="text-slate-400">ALERT STATUS:</span>
              <span className={`font-bold uppercase tracking-wider ${criticalAlert ? 'text-red-500' : (activeAlert ? 'text-amber-500' : 'text-emerald-400')}`}>
                {forecast.status}
              </span>
            </div>
          </div>

          {/* Catalog stats */}
          <div className="glass-panel p-5 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Nowcasting Stats</span>
              <span className="text-blue-500 bg-blue-500/10 p-1.5 rounded-md border border-blue-500/20">
                <FileText className="h-4 w-4" />
              </span>
            </div>
            <div className="my-4 flex items-center justify-between">
              <div>
                <div className="text-3xl font-bold font-mono tracking-tight text-slate-100">
                  {catalog.length}
                </div>
                <div className="text-xs text-slate-500 mt-1">Flares Registered In Catalog</div>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-400 block uppercase mb-1">Current State</span>
                <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                  forecast.nowcast_class === 'X-Class' ? 'bg-red-500/15 text-red-500 border border-red-500/30' :
                  forecast.nowcast_class === 'M-Class' ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30' :
                  forecast.nowcast_class === 'C-Class' ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' :
                  'bg-slate-800 text-slate-400 border border-slate-700'
                }`}>
                  {forecast.nowcast_class || "Quiet"}
                </span>
              </div>
            </div>
            <div className="text-xs text-slate-400 border-t border-slate-900 pt-2 flex items-center justify-between">
              <span>X-Class: {catalog.filter(e => e.goes_class === 'X-Class').length}</span>
              <span>M-Class: {catalog.filter(e => e.goes_class === 'M-Class').length}</span>
              <span>C-Class: {catalog.filter(e => e.goes_class === 'C-Class').length}</span>
            </div>
          </div>

        </div>

        {/* Real-time Telemetry Plotly Chart */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-amber-500" />
              <h2 className="font-semibold text-slate-200">Real-Time Light Curves & Smoothing Filters</h2>
            </div>
            <div className="text-xs text-slate-500 flex items-center space-x-4">
              <span className="flex items-center"><span className="h-2 w-2 rounded-full bg-amber-500 mr-1.5" />SoLEXS smoothed</span>
              <span className="flex items-center"><span className="h-2 w-2 rounded-full bg-red-500 mr-1.5" />HEL1OS smoothed</span>
              <span className="flex items-center"><span className="h-2.5 w-2.5 bg-slate-700/50 border border-dashed border-slate-500/50 mr-1.5" />Telemetry gaps interpolated</span>
            </div>
          </div>
          <div ref={chartRef} className="w-full h-[450px]" />
        </div>

        {/* Bottom section: FITS Processing Dropzone and Nowcasting Catalog */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* FITS Upload Processing */}
          <div className="glass-panel p-5 flex flex-col">
            <div className="flex items-center space-x-2 mb-4">
              <UploadCloud className="h-5 w-5 text-blue-400" />
              <h2 className="font-semibold text-slate-200">Offline FITS Processing</h2>
            </div>
            
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Upload raw ISRO Level-1 `.fits` files from SoLEXS and HEL1OS to synchronise counts and extract solar flare peaks immediately.
            </p>

            <form onSubmit={handleUpload} className="space-y-4 flex-1 flex flex-col justify-between">
              <div className="space-y-3">
                
                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase mb-1">SoLEXS FITS File (SXR)</label>
                  <input 
                    type="file" 
                    accept=".fits" 
                    onChange={(e) => setSolexsFile(e.target.files[0])}
                    className="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 bg-slate-900/50 border border-slate-800 rounded-lg p-1.5 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase mb-1">HEL1OS FITS File (HXR)</label>
                  <input 
                    type="file" 
                    accept=".fits" 
                    onChange={(e) => setHel1osFile(e.target.files[0])}
                    className="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 bg-slate-900/50 border border-slate-800 rounded-lg p-1.5 focus:outline-none"
                  />
                </div>

              </div>

              {uploadError && (
                <div className="p-3 text-xs rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 mt-3">
                  {uploadError}
                </div>
              )}

              {uploadResult && (
                <div className="p-3 text-xs rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 mt-3">
                  <strong>Success:</strong> Found {uploadResult.flares_detected_count} flares. Added to catalog.
                </div>
              )}

              <button
                type="submit"
                disabled={uploading || !solexsFile || !hel1osFile}
                className="w-full mt-4 py-2 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 font-semibold text-sm transition flex items-center justify-center space-x-2"
              >
                {uploading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Processing FITS...</span>
                  </>
                ) : (
                  <>
                    <Layers className="h-4 w-4" />
                    <span>Process FITS Pair</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Master Nowcasting Catalog Table */}
          <div className="glass-panel p-5 lg:col-span-2 flex flex-col h-[380px]">
            <div className="flex items-center space-x-2 mb-3">
              <FileText className="h-5 w-5 text-amber-500" />
              <h2 className="font-semibold text-slate-200">Master Nowcasting Catalog</h2>
            </div>
            
            <div className="flex-1 overflow-y-auto border border-slate-800 rounded-lg bg-slate-900/10">
              <table className="w-full text-left border-collapse text-xs">
                <thead className="bg-slate-950 text-slate-400 sticky top-0">
                  <tr>
                    <th className="py-2.5 px-4 font-semibold">Instrument</th>
                    <th className="py-2.5 px-4 font-semibold">Peak Time (UTC)</th>
                    <th className="py-2.5 px-4 font-semibold">Duration</th>
                    <th className="py-2.5 px-4 font-semibold">Peak Flux</th>
                    <th className="py-2.5 px-4 font-semibold text-right">GOES Class</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900 font-mono">
                  {catalog.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="py-8 text-center text-slate-500">
                        No flare events detected yet.
                      </td>
                    </tr>
                  ) : (
                    catalog.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40">
                        <td className="py-2 px-4 text-slate-300">{item.instrument}</td>
                        <td className="py-2 px-4 text-slate-400">
                          {item.peak_time.replace("T", " ").replace("Z", "")}
                        </td>
                        <td className="py-2 px-4 text-slate-400">
                          {Math.round(item.duration_seconds || item.duration || 0)}s
                        </td>
                        <td className="py-2 px-4 text-slate-400 font-semibold">
                          {item.peak_flux.toExponential(3)}
                        </td>
                        <td className="py-2 px-4 text-right">
                          <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                            item.goes_class.includes('X-Class') ? 'bg-red-500/15 text-red-500 border border-red-500/30' :
                            item.goes_class.includes('M-Class') ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30' :
                            'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {item.goes_class}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 px-6 text-center text-xs text-slate-600 mt-auto">
        ISRO Aditya-L1 Heliophysics Flare Nowcasting & Forecasting Pipeline · Ground Station Telemetry Segment
      </footer>
    </div>
  );
}

export default App;
