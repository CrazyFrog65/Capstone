import requests
import io
import base64
import os
import logging

try:
    from serpapi import GoogleSearch
except Exception:
    # serpapi might not be installed; we'll fallback to the HTTP API
    GoogleSearch = None


# You said you want the key hard-coded for your testing; keep it but consider using env var in future.
SERPAPI_KEY = '828a66e468b5de3e2ba626d246d665513d33cba39937688119b261354165dddc'


def upload_to_catbox_bytes(image_bytes, filename="upload.png"):
    """Upload image bytes to Catbox (simple file upload) and return a public URL.
    image_bytes: bytes
    Returns: URL string or raises Exception
    """
    url = "https://catbox.moe/user/api.php"
    files = {"fileToUpload": (filename, image_bytes)}
    data = {"reqtype": "fileupload"}
    r = requests.post(url, data=data, files=files, timeout=30)
    if r.status_code == 200:
        return r.text.strip()
    else:
        raise Exception(f"Catbox upload failed: {r.status_code} {r.text}")


def reverse_image_search(image_url, api_key=None, max_results=5):
    """Use SerpAPI (Google Lens) to fetch visual matches for a given image URL.
    Returns a list of matches with title, link, thumbnail (when available).
    If SerpAPI is not available or api_key missing, raises an informative Exception.
    """
    # If api_key not provided, fall back to hard-coded key (for your testing)
    if api_key is None:
        api_key = SERPAPI_KEY
    if not api_key:
        raise RuntimeError("SERPAPI_KEY is not configured. Set the key or pass it into reverse_image_search.")

    # First try the serpapi client if installed, otherwise use a direct HTTP fallback to the SerpAPI REST endpoint
    formatted = []
    try:
        if GoogleSearch is not None:
            params = {
                "engine": "google_lens",
                "url": image_url,
                "api_key": api_key
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            matches = results.get("visual_matches", [])
        else:
            # HTTP fallback to SerpAPI REST endpoint
            resp = requests.get("https://serpapi.com/search.json",
                                params={"engine": "google_lens", "url": image_url, "api_key": api_key},
                                timeout=30)
            resp.raise_for_status()
            results = resp.json()
            matches = results.get("visual_matches", [])

        for item in matches[:max_results]:
            formatted.append({
                "title": item.get("title") or "",
                "link": item.get("link") or item.get("source_url") or "",
                "thumbnail": item.get("thumbnail") or item.get("thumbnail_url") or ""
            })
        return formatted
    except Exception as e:
        # Raise runtime error for the caller (the Flask endpoint will catch and return it in JSON)
        logging.exception("Reverse image search failed")
        raise RuntimeError(f"Reverse image search failed: {e}")


def pil_image_to_data_url(pil_img, format="PNG"):
    """Convert a PIL image to a data URL (base64) for embedding in the frontend response.
    Returns: data:image/png;base64,... string
    """
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode('utf-8')
    return f"data:image/{format.lower()};base64,{b64}"
