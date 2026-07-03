# Run script for Solar Flare Dashboard and API Server
Write-Host "Starting Solar Flare Telemetry FastAPI Backend..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& .venv\Scripts\python.exe src/app.py"

Write-Host "Starting React Frontend Dashboard..." -ForegroundColor Green
Set-Location -Path dashboard
npm run dev
