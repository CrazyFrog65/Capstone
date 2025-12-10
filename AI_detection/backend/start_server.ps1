# Start server script for PowerShell
# Usage: .\start_server.ps1

# Create venv if not exists
if (-not (Test-Path .venv)) {
    python -m venv .venv
}

# Activate venv
. .\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run server
python app.py
