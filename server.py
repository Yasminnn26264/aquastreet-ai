"""
Safe Street AI — Local Server
Flask server with REST API + web UI for the Safe Street AI detection workflow.
Maps model classes to road damage labels ("Pothole Crack", "Pothole", "Crack").

Usage:
    $env:ROBOFLOW_API_KEY="your_key"
    python server.py              # starts on http://localhost:5000
    python server.py --port 8080  # custom port
"""

import argparse
import base64
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Add parent dir so we can import client
sys.path.insert(0, str(Path(__file__).parent))
from client import (
    RoboflowAuthError,
    RoboflowValidationError,
    RoboflowWorkflowError,
    run_safe_street_workflow,
)

app = Flask(__name__)
logger = logging.getLogger("safe_street_server")

# Directory to store uploaded images temporarily
UPLOAD_DIR = Path(tempfile.gettempdir()) / "safe_street_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def map_detection_labels(result: dict, mode: str = "pothole_crack") -> dict:
    """
    Maps raw dataset class names ('aqua route Ai', 'safe street Ai')
    to user-facing road damage labels:
      - 'pothole_crack': Displays as 'Pothole Crack'
      - 'split': 'Pothole' (for aqua route Ai) and 'Crack' (for safe street Ai)
      - 'raw': Preserves raw model class
    """
    preds_obj = result.get("predictions")
    if not isinstance(preds_obj, dict):
        return result

    det_list = preds_obj.get("predictions")
    if not isinstance(det_list, list):
        return result

    for det in det_list:
        raw_name = det.get("class", "")
        det["raw_class"] = raw_name
        raw_lower = raw_name.lower().strip()

        if mode == "raw":
            det["display_class"] = raw_name
        elif mode == "split":
            if "aqua" in raw_lower:
                det["display_class"] = "Pothole"
            elif "safe" in raw_lower or "street" in raw_lower:
                det["display_class"] = "Crack"
            else:
                det["display_class"] = "Pothole Crack"
            det["class"] = det["display_class"]
        else:
            # Default: 'pothole_crack'
            det["display_class"] = "Pothole Crack"
            det["class"] = "Pothole Crack"

    return result


# ── Web UI ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return INDEX_HTML


# ── REST API ────────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect

    Accepts either:
      - multipart/form-data with field "image" (file upload)
      - JSON body with:
          {"image_url": "https://..."}        — HTTPS URL
          {"image_base64": "<base64-data>"}   — raw base64

    Optional fields (form or JSON):
      - confidence: float (0.0–1.0 or 0–100)
      - iou_threshold: float
      - max_detections: int
      - label_mode: 'pothole_crack' (default), 'split', or 'raw'

    Returns JSON with detection results.
    """
    try:
        image_input = None
        confidence = None
        label_mode = "pothole_crack"
        extra_params = {}

        content_type = request.content_type or ""

        if "multipart/form-data" in content_type:
            # ── File upload ──
            if "image" not in request.files:
                return jsonify({"error": "No 'image' file in upload"}), 400
            file = request.files["image"]
            if not file.filename:
                return jsonify({"error": "Empty filename"}), 400

            ext = Path(file.filename).suffix or ".jpg"
            tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
            file.save(str(tmp_path))
            image_input = str(tmp_path)

            confidence = request.form.get("confidence", type=float)
            label_mode = request.form.get("label_mode", default="pothole_crack")
            iou = request.form.get("iou_threshold", type=float)
            max_det = request.form.get("max_detections", type=int)
            if iou is not None:
                extra_params["iou_threshold"] = iou
            if max_det is not None:
                extra_params["max_detections"] = max_det

        elif "application/json" in content_type:
            # ── JSON body ──
            body = request.get_json(silent=True) or {}

            if "image_url" in body:
                image_input = body["image_url"]
            elif "image_base64" in body:
                b64 = body["image_base64"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                tmp_path.write_bytes(base64.b64decode(b64))
                image_input = str(tmp_path)
            else:
                return jsonify({"error": "Provide 'image_url' or 'image_base64'"}), 400

            confidence = body.get("confidence")
            label_mode = body.get("label_mode", "pothole_crack")
            if "iou_threshold" in body:
                extra_params["iou_threshold"] = body["iou_threshold"]
            if "max_detections" in body:
                extra_params["max_detections"] = body["max_detections"]
        else:
            return jsonify({"error": f"Unsupported Content-Type: {content_type}"}), 415

        # ── Run workflow ──
        result = run_safe_street_workflow(
            image=image_input,
            confidence=confidence,
            parameters=extra_params if extra_params else None,
            output_dir=str(UPLOAD_DIR / "outputs"),
        )

        # Apply label mapping (e.g. "Pothole Crack")
        mapped_result = map_detection_labels(result, mode=label_mode)

        return jsonify({"success": True, "data": mapped_result})

    except RoboflowAuthError as e:
        return jsonify({"error": f"Auth error: {e}"}), 401
    except RoboflowValidationError as e:
        return jsonify({"error": f"Validation error: {e}"}), 422
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except RoboflowWorkflowError as e:
        return jsonify({"error": f"Workflow error: {e}"}), 502
    except Exception as e:
        logger.exception("Unexpected error in /detect")
        return jsonify({"error": f"Internal error: {e}"}), 500
    finally:
        # Clean up temp upload (non-blocking)
        if image_input and Path(image_input).exists() and str(UPLOAD_DIR) in str(image_input):
            try:
                Path(image_input).unlink(missing_ok=True)
            except Exception:
                pass


@app.route("/health")
def health():
    has_key = bool(os.getenv("ROBOFLOW_API_KEY"))
    return jsonify({"status": "ok", "api_key_set": has_key})


# ── Inline HTML UI ──────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safe Street AI — Pothole & Crack Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 920px; margin: 0 auto; padding: 2rem 1rem; }
  .header-badge { display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.75rem; border: 1px solid rgba(56, 189, 248, 0.3); }
  h1 { text-align: center; font-size: 2rem; margin-bottom: .4rem; color: #f8fafc; font-weight: 700; }
  h1 span { color: #f59e0b; }
  .subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .95rem; }
  .card { background: #1e293b; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
  .drop-zone { border: 2px dashed #475569; border-radius: 12px; padding: 3rem 1rem; text-align: center; cursor: pointer; transition: all .2s; background: rgba(15, 23, 42, 0.4); }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #f59e0b; background: rgba(245, 158, 11, 0.08); }
  .drop-zone p { color: #cbd5e1; margin-top: .5rem; font-size: 1rem; }
  .drop-zone .hint { color: #64748b; font-size: 0.85rem; margin-top: 0.25rem; }
  .drop-zone .icon { font-size: 2.8rem; }
  input[type="file"] { display: none; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.2rem; }
  .row > * { flex: 1; min-width: 160px; }
  label { display: block; font-size: .85rem; color: #94a3b8; margin-bottom: .4rem; font-weight: 500; }
  input[type="number"], input[type="text"], select {
    width: 100%; padding: .55rem .75rem; background: #0f172a; border: 1px solid #334155;
    border-radius: 8px; color: #e2e8f0; font-size: .95rem; outline: none; transition: border-color .2s;
  }
  input[type="number"]:focus, input[type="text"]:focus, select:focus {
    border-color: #f59e0b;
  }
  .btn { display: inline-block; padding: .75rem 1.5rem; background: linear-gradient(135deg, #f59e0b, #d97706); color: #fff; border: none;
         border-radius: 10px; font-size: 1.05rem; font-weight: 600; cursor: pointer; transition: transform .15s, box-shadow .15s; width: 100%; margin-top: 1.2rem; }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3); }
  .btn:disabled { background: #475569; cursor: not-allowed; transform: none; box-shadow: none; }
  .preview-wrapper { text-align: center; margin-top: 1rem; }
  .preview-img { max-width: 100%; max-height: 340px; border-radius: 8px; display: none; border: 1px solid #334155; }
  .results { margin-top: 1.5rem; }
  .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }
  .det-box { background: #0f172a; border-radius: 10px; padding: .85rem 1.1rem; margin-top: .6rem; display: flex; justify-content: space-between; align-items: center; border-left: 4px solid #f59e0b; border: 1px solid #334155; border-left-width: 4px; }
  .det-class { font-weight: 700; color: #f59e0b; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem; }
  .det-raw { font-size: 0.75rem; color: #64748b; background: rgba(100, 116, 139, 0.15); padding: 0.15rem 0.45rem; border-radius: 4px; }
  .det-conf { color: #22c55e; font-weight: 700; font-size: 1.1rem; }
  .det-coords { color: #64748b; font-size: .82rem; margin-top: 0.2rem; }
  .error { color: #f87171; margin-top: 1rem; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2); }
  .spinner { display: none; text-align: center; padding: 2rem; }
  .spinner.active { display: block; }
  .spinner .dot { display: inline-block; width: 10px; height: 10px; background: #f59e0b; border-radius: 50%; margin: 0 4px; animation: bounce .6s infinite alternate; }
  .spinner .dot:nth-child(2) { animation-delay: .2s; }
  .spinner .dot:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { to { transform: translateY(-12px); opacity: .4; } }
  .meta { color: #94a3b8; font-size: .9rem; }
  .canvas-wrapper { position: relative; display: block; margin-top: 1.2rem; text-align: center; }
  .canvas-wrapper canvas { max-width: 100%; height: auto; border-radius: 10px; border: 1px solid #334155; }
  .or-divider { text-align: center; color: #64748b; margin: 1.2rem 0; font-size: .85rem; position: relative; }
  .summary-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(245, 158, 11, 0.15); color: #f59e0b; font-size: 0.85rem; padding: 0.25rem 0.6rem; border-radius: 6px; font-weight: 600; }
</style>
</head>
<body>
<div class="container">
  <div style="text-align: center;">
    <div class="header-badge">Road Hazard & Defect Inspection</div>
    <h1>🛡️ Safe Street <span>AI</span></h1>
    <p class="subtitle">Automated Pothole & Road Crack Detection powered by RF-DETR</p>
  </div>

  <div class="card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">🚧</div>
      <p><strong>Drop road / street image here</strong> or click to browse</p>
      <div class="hint">Supports JPG, PNG, WEBP (inspects road surfaces for potholes & cracks)</div>
    </div>
    <input type="file" id="fileInput" accept="image/*">
    <div class="preview-wrapper">
      <img class="preview-img" id="previewImg">
    </div>

    <div class="or-divider">— or inspect an image URL —</div>
    <input type="text" id="urlInput" placeholder="https://example.com/pothole-street.jpg">

    <div class="row">
      <div>
        <label>Label Display Mode</label>
        <select id="labelModeSelect">
          <option value="pothole_crack" selected>Pothole Crack</option>
          <option value="split">Pothole / Crack (Split)</option>
          <option value="raw">Raw Model Classes</option>
        </select>
      </div>
      <div>
        <label>Confidence Threshold</label>
        <input type="number" id="confInput" value="0.4" min="0" max="1" step="0.05">
      </div>
      <div>
        <label>IoU Threshold</label>
        <input type="number" id="iouInput" value="0.3" min="0" max="1" step="0.05">
      </div>
      <div>
        <label>Max Detections</label>
        <input type="number" id="maxDetInput" value="1000" min="1" step="1">
      </div>
    </div>

    <button class="btn" id="detectBtn" onclick="runDetection()">🔍 Detect Pothole / Cracks</button>
  </div>

  <div class="spinner" id="spinner"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
  <div class="error" id="errorBox" style="display:none;"></div>

  <div class="card results" id="resultsCard" style="display:none;">
    <div class="results-header">
      <h3 style="font-size: 1.2rem;">Inspection Findings</h3>
      <div id="summaryBadge"></div>
    </div>
    <div class="meta" id="metaInfo"></div>
    <div class="canvas-wrapper" id="canvasWrapper" style="display:none;">
      <canvas id="annotCanvas"></canvas>
    </div>
    <div id="detList" style="margin-top: 1rem;"></div>
  </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
let selectedFile = null;

['dragenter','dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); }));
dropZone.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) { selectedFile = ev.dataTransfer.files[0]; showPreview(selectedFile); } });
fileInput.addEventListener('change', () => { if (fileInput.files.length) { selectedFile = fileInput.files[0]; showPreview(selectedFile); } });

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => { previewImg.src = e.target.result; previewImg.style.display = 'inline-block'; };
  reader.readAsDataURL(file);
  document.getElementById('urlInput').value = '';
}

async function runDetection() {
  const btn = document.getElementById('detectBtn');
  const spinner = document.getElementById('spinner');
  const errorBox = document.getElementById('errorBox');
  const resultsCard = document.getElementById('resultsCard');
  const urlInput = document.getElementById('urlInput').value.trim();
  const labelMode = document.getElementById('labelModeSelect').value;

  errorBox.style.display = 'none';
  errorBox.textContent = '';
  resultsCard.style.display = 'none';
  btn.disabled = true;
  spinner.classList.add('active');

  try {
    let res;
    const conf = parseFloat(document.getElementById('confInput').value) || 0.4;
    const iou = parseFloat(document.getElementById('iouInput').value) || 0.3;
    const maxDet = parseInt(document.getElementById('maxDetInput').value) || 1000;

    if (selectedFile) {
      const form = new FormData();
      form.append('image', selectedFile);
      form.append('confidence', conf);
      form.append('iou_threshold', iou);
      form.append('max_detections', maxDet);
      form.append('label_mode', labelMode);
      res = await fetch('/detect', { method: 'POST', body: form });
    } else if (urlInput) {
      // Also preview the URL image
      previewImg.src = urlInput;
      previewImg.style.display = 'inline-block';
      res = await fetch('/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: urlInput, confidence: conf, iou_threshold: iou, max_detections: maxDet, label_mode: labelMode }),
      });
    } else {
      errorBox.textContent = 'Please select a road image file or enter an image URL.';
      errorBox.style.display = 'block';
      return;
    }

    const json = await res.json();
    if (!res.ok || json.error) {
      errorBox.textContent = json.error || 'Request failed';
      errorBox.style.display = 'block';
      return;
    }
    renderResults(json.data);
  } catch (e) {
    errorBox.textContent = 'Network error: ' + e.message;
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.classList.remove('active');
  }
}

function renderResults(data) {
  const resultsCard = document.getElementById('resultsCard');
  const metaInfo = document.getElementById('metaInfo');
  const detList = document.getElementById('detList');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const summaryBadge = document.getElementById('summaryBadge');

  resultsCard.style.display = 'block';
  detList.innerHTML = '';

  const preds = data.predictions || {};
  const imgMeta = preds.image || {};
  const detections = preds.predictions || [];

  metaInfo.textContent = `Image dimensions: ${imgMeta.width || '?'} × ${imgMeta.height || '?'} px`;

  if (detections.length === 0) {
    summaryBadge.innerHTML = '<span class="summary-pill" style="color:#22c55e; background:rgba(34,197,94,0.15);">✅ Road Surface Clear</span>';
    detList.innerHTML = '<p style="color:#94a3b8;margin-top:.5rem;">No potholes or cracks detected above the confidence threshold.</p>';
    canvasWrapper.style.display = 'none';
    return;
  }

  summaryBadge.innerHTML = `<span class="summary-pill">⚠️ ${detections.length} Pothole Crack(s) Detected</span>`;

  // Color mapping for road hazards (amber/red for potholes and cracks)
  const colors = ['#f59e0b', '#ef4444', '#ec4899', '#38bdf8', '#22c55e'];
  const classColors = {};
  let ci = 0;

  detections.forEach((d, i) => {
    const displayLabel = d.display_class || d.class || "Pothole Crack";
    if (!classColors[displayLabel]) classColors[displayLabel] = colors[ci++ % colors.length];
    const c = classColors[displayLabel];

    const rawBadge = d.raw_class && d.raw_class !== displayLabel
      ? `<span class="det-raw" title="Underlying model class">${d.raw_class}</span>`
      : '';

    detList.innerHTML += `
      <div class="det-box" style="border-left-color:${c}">
        <div>
          <div class="det-class" style="color:${c}">
            <span>⚠️ ${displayLabel}</span>
            ${rawBadge}
          </div>
          <div class="det-coords">Coordinates: center=(${d.x?.toFixed(1)}, ${d.y?.toFixed(1)}) • size=${d.width?.toFixed(1)} × ${d.height?.toFixed(1)} px</div>
        </div>
        <span class="det-conf">${(d.confidence * 100).toFixed(1)}%</span>
      </div>`;
  });

  // Draw bounding boxes on canvas
  const img = previewImg;
  if (img && img.src && detections.length > 0) {
    const canvas = document.getElementById('annotCanvas');
    const ctx = canvas.getContext('2d');
    const drawImg = new Image();
    drawImg.crossOrigin = 'anonymous';
    drawImg.onload = () => {
      canvas.width = drawImg.naturalWidth;
      canvas.height = drawImg.naturalHeight;
      ctx.drawImage(drawImg, 0, 0);

      detections.forEach(d => {
        const displayLabel = d.display_class || d.class || "Pothole Crack";
        const c = classColors[displayLabel] || '#f59e0b';
        const bx = d.x - d.width / 2;
        const by = d.y - d.height / 2;

        // Bounding box border
        ctx.strokeStyle = c;
        ctx.lineWidth = Math.max(3, Math.round(canvas.width / 350));
        ctx.strokeRect(bx, by, d.width, d.height);

        // Bounding box subtle fill
        ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
        ctx.fillRect(bx, by, d.width, d.height);

        // Tag label header
        const labelText = `${displayLabel} ${(d.confidence * 100).toFixed(0)}%`;
        const fontSize = Math.max(14, Math.round(canvas.width / 45));
        ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
        const tw = ctx.measureText(labelText).width;
        const th = fontSize + 8;

        ctx.fillStyle = c;
        ctx.fillRect(bx, Math.max(0, by - th), tw + 12, th);

        ctx.fillStyle = '#0f172a';
        ctx.fillText(labelText, bx + 6, Math.max(th - 5, by - 6));
      });
      canvasWrapper.style.display = 'block';
    };
    drawImg.src = img.src;
  }
}
</script>
</body>
</html>
"""


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Street AI detection server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.getenv("ROBOFLOW_API_KEY"):
        print("\n⚠️  WARNING: ROBOFLOW_API_KEY is not set!")
        print("   Set it with:  $env:ROBOFLOW_API_KEY=\"your_key_here\"\n")

    print(f"\n🛡️  Safe Street AI server starting on http://{args.host}:{args.port}")
    print(f"   Open in browser: http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
