# Roboflow Workflow Integration: Safe Street AI

This module integrates the Roboflow Workflow **safe-street-ai-1-rfdetr-small-t1 vsafe-street-ai-1-rfdetr-small-t1 Logic** into your project.

## Workflow Details

| Field | Value |
|-------|-------|
| **Workflow Name** | `safe-street-ai-1-rfdetr-small-t1 vsafe-street-ai-1-rfdetr-small-t1 Logic` |
| **Workspace Slug** | `yasmins-workspace-n6tkr` |
| **Workflow Slug** | `safe-street-ai-1-rfdetr-small-t1-vsafe-street-ai-1-rfdetr-small-t1-logic` |
| **Endpoint** | `https://serverless.roboflow.com/yasmins-workspace-n6tkr/workflows/safe-street-ai-1-rfdetr-small-t1-vsafe-street-ai-1-rfdetr-small-t1-logic` |
| **Model** | RF-DETR Small — object detection (`yasmins-workspace-n6tkr/safe-street-ai-1-rfdetr-small-t1`) |
| **Input** | `image` (HTTPS URL, local file path, or base64) |
| **Output key** | `predictions` — detection boxes with class, confidence, and bounding box |
| **Known classes** | `aqua route Ai`, `safe street Ai` |

---

## 1. Environment Setup

### Set your Roboflow API Key

Obtain your API key from [Roboflow API Settings](https://app.roboflow.com/settings/api).

**PowerShell:**
```powershell
$env:ROBOFLOW_API_KEY="your_api_key_here"
```

**Bash:**
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

### Dependencies

You can install the official `inference-sdk`:
```bash
pip install inference-sdk
```
> **Note:** If `inference-sdk` is not installed, the client automatically falls back to standard-library REST requests (zero external dependencies).

---

## 2. Usage

### Python

```python
from client import run_safe_street_workflow

# Run inference on a remote image
results = run_safe_street_workflow(
    "https://example.com/street-photo.jpg",
    confidence=0.4,
)

preds = results["predictions"]
print(f"Image size: {preds['image']}")
for det in preds["predictions"]:
    print(f"  {det['class']}  conf={det['confidence']:.2f}  "
          f"box=({det['x']}, {det['y']}, {det['width']}, {det['height']})")

# Run on a local file with custom parameters
results = run_safe_street_workflow(
    "./street_cam.png",
    confidence=0.5,
    parameters={"iou_threshold": 0.3, "max_detections": 100},
    output_dir="./saved_predictions",
)
```

### TypeScript / Node.js

```typescript
import { runSafeStreetWorkflow } from './client';

const results = await runSafeStreetWorkflow('./street_cam.png', {
  confidence: 0.5,
  parameters: { iou_threshold: 0.3 },
  outputDir: './saved_predictions',
});

const preds = results.predictions;
console.log(`Image size: ${preds.image.width}x${preds.image.height}`);
for (const det of preds.predictions) {
  console.log(`  ${det.class}  conf=${det.confidence}  box=(${det.x}, ${det.y}, ${det.width}, ${det.height})`);
}
```

### Workflow Parameters

These parameters can be passed via the `parameters` dict:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | float | 0.4 | Minimum detection confidence |
| `iou_threshold` | float | 0.3 | IoU threshold for NMS |
| `class_agnostic_nms` | bool | false | Use class-agnostic NMS |
| `max_detections` | int | 1000 | Maximum detections to return |

### Confidence Threshold Behavior

- **Server-side:** Forwarded to the workflow as `confidence` and `confidence_threshold` parameters.
- **Client-side:** Filters out detection records below the threshold after response.
- **Flexible formats:** Accepts decimals (e.g. `0.65`) or percentages (e.g. `65`). Invalid values raise a typed `RoboflowValidationError`.

---

## 3. Web UI & Local Server

Run the interactive local detection dashboard:

```bash
# PowerShell
$env:ROBOFLOW_API_KEY="your_api_key_here"
python server.py --port 5000
```

Open your browser at `http://localhost:5000` to:
- Drag & drop or upload road inspection images
- Detect road hazards with bounding box overlay visualization
- Filter by confidence threshold, IoU threshold, and max detections
- Select label display modes: `Pothole Crack` (default), `Pothole / Crack (Split)`, or `Raw Model Classes`

### REST API Endpoint

`POST /detect` accepts `multipart/form-data` (file upload) or `application/json`:

```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/street.jpg", "confidence": 0.4, "label_mode": "pothole_crack"}'
```

---

## 4. Response Format

The API returns a dict with key `predictions`:

```json
{
  "predictions": {
    "image": { "width": 720, "height": 1280 },
    "predictions": [
      {
        "x": 323.0, "y": 811.5,
        "width": 640.0, "height": 911.0,
        "confidence": 0.912,
        "class_id": 0,
        "class": "aqua route Ai",
        "detection_id": "...",
        "parent_id": "image"
      }
    ]
  }
}
```

---

## 5. Running Smoke Tests

```bash
python smoke_test.py
```

Offline tests (validation, filtering) always run. Live tests require `ROBOFLOW_API_KEY` to be set.

---

## 6. Architecture

- **`client.py`** — Python client with SDK and REST fallback
- **`client.ts`** — TypeScript client (REST)
- **`smoke_test.py`** — Offline + live tests

Auth is always via `Authorization: Bearer` header (never query string or body).
API key is always read from `ROBOFLOW_API_KEY` environment variable (never hardcoded).
