# AI Detection — Frontend & Backend Integration

This folder contains a lightweight frontend (`index.html`) and a backend (`backend/`) that loads your PyTorch model for inference.

Quick start
1. Ensure model weights (`best_model_fusion2.pth` for the Fusion Model or `best_model.pth` for ResNet-only) are present in the repository root or in the `backend` directory.

2. Setup and run backend (PowerShell commands):
```powershell
# Make a virtual environment and activate it
python -m venv .venv
. .\venv\Scripts\Activate.ps1

# Install requirements
pip install -r backend\requirements.txt

# Run the API server
python backend\app.py
```

3. Open `index.html` in a browser (double-click or serve via a static server) and test the UI.

If your browser blocks making fetch calls from file:// pages, serve the folder to a local port:
```powershell
# From the AI_detection folder
python -m http.server 8000
# Then open http://127.0.0.1:8000 in your browser
```

Notes
- The Flask server runs on `http://127.0.0.1:5000` by default. If you need to host on a different host/port, set the `FLASK_HOST` and `FLASK_PORT` environment variables before running `app.py`.
- For cross-origin requests from other origins, `flask_cors` is used so the frontend can call the backend directly.
- If the Fusion model weights are not present, the backend will attempt to use ResNet weights. If no weights are provided, the model will still run but predictions will be from untrained random weights (not recommended).

Optional: Reverse Image Search
- The backend supports reverse image search using SerpAPI (Google Lens). To enable reverse search, set `SERPAPI_KEY` in your environment before running the backend:
	```powershell
	$env:SERPAPI_KEY = 'your_serpapi_api_key'
	python backend\app.py
	```
- When the model predicts `real`, the server will perform a reverse image search and return a list of matches (title, link, thumbnail) to the frontend.

If you'd like, I can add a small Dockerfile for quick containerization or add a small Node static server (e.g. `http-server`) for easier testing.
