"""
Roboflow Workflow Client for:
Workflow: safe-street-ai-1-rfdetr-small-t1 vsafe-street-ai-1-rfdetr-small-t1 Logic
Workspace: yasmins-workspace-n6tkr
Workflow ID: safe-street-ai-1-rfdetr-small-t1-vsafe-street-ai-1-rfdetr-small-t1-logic

Inputs (from workflow spec):
  - image (InferenceImage)

Parameters (from inner workflow "safe-street-ai"):
  - confidence (float, default 0.4)
  - iou_threshold (float, default 0.3)
  - class_agnostic_nms (bool, default False)
  - max_detections (int, default 1000)

Output keys (from live response):
  - predictions: {
      "image": {"width": int, "height": int},
      "predictions": [{"x", "y", "width", "height", "confidence", "class_id", "class", "detection_id", "parent_id"}, ...]
    }

Known classes: "aqua route Ai", "safe street Ai"
"""

import os
import time
import json
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.error

logger = logging.getLogger("safe_street_ai")

# ── Configuration ───────────────────────────────────────────────────────
WORKSPACE_NAME = "yasmins-workspace-n6tkr"
WORKFLOW_ID = (
    "safe-street-ai-1-rfdetr-small-t1-vsafe-street-ai-1-rfdetr-small-t1-logic"
)
SERVERLESS_URL = "https://serverless.roboflow.com"
WORKFLOW_ENDPOINT = f"{SERVERLESS_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"


# ── Typed Exceptions ───────────────────────────────────────────────────
class RoboflowWorkflowError(Exception):
    """Base exception for Roboflow Workflow operations."""


class RoboflowAuthError(RoboflowWorkflowError):
    """Raised when authentication fails (missing or invalid API key)."""


class RoboflowTimeoutError(RoboflowWorkflowError):
    """Raised when an inference request times out."""


class RoboflowValidationError(RoboflowWorkflowError):
    """Raised when request inputs/parameters are invalid."""


# ── Helpers ─────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowAuthError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Get your key at https://app.roboflow.com/settings/api"
        )
    return api_key


def _validate_confidence(confidence: Optional[Union[float, int]]) -> Optional[float]:
    """
    Normalize confidence to [0.0, 1.0].
    Accepts 0.0–1.0 (fraction) or 1.0–100.0 (percent).
    """
    if confidence is None:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RoboflowValidationError(
            f"Invalid confidence value: {confidence!r}. "
            "Must be a number in [0.0, 1.0] or percentage [0, 100]."
        )
    val = float(confidence)
    if val < 0.0:
        raise RoboflowValidationError(f"Confidence cannot be negative. Got: {val}")
    if val > 100.0:
        raise RoboflowValidationError(
            f"Confidence cannot exceed 1.0 (or 100%). Got: {val}"
        )
    if val > 1.0:
        val /= 100.0
    return val


def _save_base64_image(b64_string: str, output_path: Union[str, Path]) -> str:
    """Decode base64 image data and write to disk immediately."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(image_bytes)
    return str(out_file.resolve())


def _filter_predictions(
    item: Any, min_confidence: Optional[float] = None
) -> Any:
    """
    Recursively cleans prediction outputs:
    - Filters out records whose confidence/score is below min_confidence
    - Strips heavy raw segmentation `points` if present
    """
    if isinstance(item, list):
        filtered = []
        for elem in item:
            cleaned = _filter_predictions(elem, min_confidence)
            if cleaned is not None:
                filtered.append(cleaned)
        return filtered

    if isinstance(item, dict):
        conf = item.get("confidence") or item.get("score")
        if (
            min_confidence is not None
            and isinstance(conf, (int, float))
            and conf < min_confidence
        ):
            return None

        cleaned_dict = {}
        for k, v in item.items():
            if k == "points":
                continue  # strip heavy polygon points
            cleaned_v = _filter_predictions(v, min_confidence)
            if cleaned_v is not None:
                cleaned_dict[k] = cleaned_v
        return cleaned_dict

    return item


def _process_outputs(
    raw_outputs: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    image_prefix: str = "output",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Safely process workflow outputs:
    - Decodes any base64 image blobs to disk (never kept in memory)
    - Filters detection records by confidence threshold
    - Strips heavy segmentation points
    - Preserves all output keys dynamically
    """
    processed: Dict[str, Any] = {}
    out_dir = Path(output_dir) if output_dir else Path("./output_images")

    for key, value in raw_outputs.items():
        # base64 image string
        if isinstance(value, str) and (
            value.startswith("data:image/")
            or (len(value) > 1000 and value.isalnum())
        ):
            try:
                dest = out_dir / f"{image_prefix}_{key}.jpg"
                saved_path = _save_base64_image(value, dest)
                processed[key] = {"type": "saved_image", "path": saved_path}
                continue
            except Exception as e:
                logger.warning(f"Failed to decode image for key '{key}': {e}")
                processed[key] = "<unprocessed_binary_data>"
                continue

        # dict with explicit base64 payload
        if isinstance(value, dict) and value.get("type") == "base64" and "value" in value:
            dest = out_dir / f"{image_prefix}_{key}.jpg"
            saved_path = _save_base64_image(value["value"], dest)
            processed[key] = {"type": "saved_image", "path": saved_path}
            continue

        # everything else: filter detections and strip heavy fields
        processed[key] = _filter_predictions(value, min_confidence=confidence)

    return processed


# ── SDK Client ──────────────────────────────────────────────────────────
def run_safe_street_workflow_sdk(
    image: Union[str, Path],
    confidence: Optional[Union[float, int]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """
    Run the workflow using the official inference-sdk (InferenceHTTPClient).
    """
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError:
        raise ImportError(
            "inference-sdk not installed. Install with 'pip install inference-sdk' "
            "or use run_safe_street_workflow_rest() instead."
        )

    conf_val = _validate_confidence(confidence)
    workflow_params: Dict[str, Any] = dict(parameters) if parameters else {}
    if conf_val is not None:
        workflow_params.setdefault("confidence", conf_val)
        workflow_params.setdefault("confidence_threshold", conf_val)

    api_key = _get_api_key()
    client = InferenceHTTPClient(api_url=SERVERLESS_URL, api_key=api_key)
    client.configure(InferenceConfiguration(api_key_transport="header"))

    # Resolve image input
    image_str = str(image)
    if not image_str.startswith(("http://", "https://")):
        image_path = Path(image_str)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_str}")
        image_input = str(image_path.resolve())
    else:
        image_input = image_str

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            results = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": image_input},
                parameters=workflow_params,
            )

            # SDK returns a list (one entry per input image)
            if isinstance(results, list) and len(results) > 0:
                raw = results[0]
            elif isinstance(results, dict):
                # Serverless REST wraps in {"outputs": [...]}
                raw = results.get("outputs", [results])[0] if "outputs" in results else results
            else:
                raw = {"raw": results}

            return _process_outputs(raw, output_dir=output_dir, confidence=conf_val)

        except Exception as e:
            last_exc = e
            if "401" in str(e) or "Unauthorized" in str(e):
                raise RoboflowAuthError(f"Authentication failed: {e}") from e
            if attempt < max_retries:
                wait = backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

    raise RoboflowWorkflowError(
        f"Workflow failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


# ── REST Client ─────────────────────────────────────────────────────────
def run_safe_street_workflow_rest(
    image: Union[str, Path],
    confidence: Optional[Union[float, int]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """
    Run the workflow via direct REST call (zero external dependencies).
    """
    conf_val = _validate_confidence(confidence)
    workflow_params: Dict[str, Any] = dict(parameters) if parameters else {}
    if conf_val is not None:
        workflow_params.setdefault("confidence", conf_val)
        workflow_params.setdefault("confidence_threshold", conf_val)

    api_key = _get_api_key()

    # Build image payload
    image_str = str(image)
    if image_str.startswith("https://"):
        image_payload = {"type": "url", "value": image_str}
    elif image_str.startswith("http://"):
        raise RoboflowValidationError(
            "Image URL must use HTTPS (plain HTTP is rejected by Roboflow)."
        )
    else:
        path_obj = Path(image_str)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_str}")
        b64_content = base64.b64encode(path_obj.read_bytes()).decode("utf-8")
        image_payload = {"type": "base64", "value": b64_content}

    payload: Dict[str, Any] = {"inputs": {"image": image_payload}}
    if workflow_params:
        payload["parameters"] = workflow_params

    req_data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                WORKFLOW_ENDPOINT, data=req_data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

                # REST endpoint returns {"outputs": [{...}], "profiler_trace": ...}
                if isinstance(data, dict) and "outputs" in data:
                    outputs_list = data["outputs"]
                    raw = outputs_list[0] if outputs_list else {}
                elif isinstance(data, list) and len(data) > 0:
                    raw = data[0]
                elif isinstance(data, dict):
                    raw = data
                else:
                    raw = {"raw": data}

                return _process_outputs(
                    raw, output_dir=output_dir, confidence=conf_val
                )

        except urllib.error.HTTPError as http_err:
            last_exc = http_err
            if http_err.code in (401, 403):
                raise RoboflowAuthError(
                    f"Authentication failed (HTTP {http_err.code}): "
                    f"{http_err.read().decode('utf-8')}"
                ) from http_err
            if http_err.code >= 500 and attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue
            raise RoboflowWorkflowError(
                f"HTTP error {http_err.code}: {http_err.read().decode('utf-8')}"
            ) from http_err

        except urllib.error.URLError as url_err:
            last_exc = url_err
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue
            raise RoboflowTimeoutError(
                f"Connection/Timeout error: {url_err}"
            ) from url_err

        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

    raise RoboflowWorkflowError(
        f"Workflow failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


# ── Main Entry Point ────────────────────────────────────────────────────
def run_safe_street_workflow(
    image: Union[str, Path],
    confidence: Optional[Union[float, int]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    max_retries: int = 3,
    prefer_sdk: bool = True,
) -> Dict[str, Any]:
    """
    Run the Safe Street AI object-detection workflow.

    Args:
        image: Local file path or HTTPS URL of the image to analyze.
        confidence: Optional confidence threshold (0.0–1.0 or 0–100%).
            Filters out detections below this score and also forwarded
            to the workflow's server-side parameters.
        parameters: Optional dict of additional workflow parameters:
            - confidence (float, default 0.4)
            - iou_threshold (float, default 0.3)
            - class_agnostic_nms (bool, default False)
            - max_detections (int, default 1000)
        output_dir: Directory for saving any base64 image outputs.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient errors.
        prefer_sdk: If True, tries inference-sdk first, then falls back to REST.

    Returns:
        Dict with output key "predictions" containing:
        {
            "image": {"width": int, "height": int},
            "predictions": [
                {
                    "x": float, "y": float, "width": float, "height": float,
                    "confidence": float, "class_id": int, "class": str,
                    "detection_id": str, "parent_id": str
                }, ...
            ]
        }
    """
    if prefer_sdk:
        try:
            return run_safe_street_workflow_sdk(
                image=image,
                confidence=confidence,
                parameters=parameters,
                output_dir=output_dir,
                timeout=timeout,
                max_retries=max_retries,
            )
        except ImportError:
            logger.info("inference-sdk not installed, falling back to REST client.")

    return run_safe_street_workflow_rest(
        image=image,
        confidence=confidence,
        parameters=parameters,
        output_dir=output_dir,
        timeout=timeout,
        max_retries=max_retries,
    )
