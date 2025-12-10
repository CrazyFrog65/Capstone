# AI Detection — Backend

This backend provides a small Flask API to load your trained model and run predictions from the frontend.

How it works
- `/predict` — POST endpoint which accepts either a multipart `image` file or JSON `{ "url": "https://..." }` to download and predict.

Requirements
- Python 3.8+
- Install dependencies (see below)

Quick setup and run
```powershell
# (Recommended) Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
# Make sure your model files are in the repo root (e.g., best_model_fusion2.pth or best_model.pth)
python app.py
```

Notes
- If `best_model_fusion2.pth` exists, it'll be loaded as the fusion model (ResNet18 + EfficientNetB0). Otherwise, the app will try to load `best_model.pth` into ResNet18.
- If you want to run on CPU, the app will automatically use the CPU if a CUDA-enabled GPU is not present.

- Reverse Image Search (optional):
	- If the prediction is `real`, the server will attempt to perform a reverse image search using SerpAPI (Google Lens) to provide visual matches.
	- To enable this feature, set the `SERPAPI_KEY` environment variable to your SerpAPI API key prior to running the server. Example:
		```powershell
		$env:SERPAPI_KEY = 'your_serpapi_key_here'
		python app.py
		```
	- The backend uploads the image to Catbox.moe to get a public URL for the reverse search; Catbox automatically deletes uploads over time (note: check their terms/usage).

CORS
- `flask_cors` is used so the React/HTML frontend can call it from a different origin during development.

If you want a single-file binary, containerization (Docker) or deployment instructions can be added.

Test endpoints with curl
```powershell
# 1) With file upload
curl -X POST -F "image=@C:\\path\\to\\fake1.png" http://127.0.0.1:5000/predict

# 2) With URL
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/image.jpg"}' http://127.0.0.1:5000/predict
```
