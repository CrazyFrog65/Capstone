from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, predict_image
from search import upload_to_catbox_bytes, reverse_image_search, pil_image_to_data_url
from PIL import Image
import io
import os
import requests

app = Flask(__name__)
CORS(app)

# Load model once
MODEL, CLASS_NAMES = load_model()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    # Accept either file upload (multipart/form-data) or JSON with {"url": "http..."}
    public_url = None
    input_was_url = False
    if 'image' in request.files:
        file = request.files['image']
        try:
            img = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Invalid image uploaded. {e}"}), 400
    else:
        data = request.get_json(silent=True)
        if not data or 'url' not in data:
            return jsonify({"error": "No image file or URL provided."}), 400
        image_url = data['url']
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            public_url = image_url
            input_was_url = True
        except Exception as e:
            return jsonify({"error": f"Failed to download image from URL: {e}"}), 400

    # Run prediction
    try:
        result = predict_image(MODEL, img, CLASS_NAMES)
        # embed original image so frontend can display it
        try:
            result['original_image'] = pil_image_to_data_url(img)
        except Exception:
            # not critical, proceed without embedded image
            pass

        # If predicted REAL, try reverse image search
        if result.get('class') == 'real':
            try:
                # For URL inputs, use the provided URL; otherwise upload to Catbox
                if not (input_was_url and public_url):
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    public_url = upload_to_catbox_bytes(img_bytes)

                if public_url:
                    results = reverse_image_search(public_url)
                    result['reverse_results'] = results
                    result['reverse_source_url'] = public_url
            except Exception as e:
                # include a message about reverse search failure but still return predictions
                result['reverse_error'] = str(e)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Read host and port from env or default
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    print(f"Starting server on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
