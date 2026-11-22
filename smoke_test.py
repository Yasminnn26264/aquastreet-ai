"""
Smoke tests for the Safe Street AI Roboflow workflow client.

Asserts:
1. Missing API key → RoboflowAuthError
2. Plain http:// URL → RoboflowValidationError
3. Confidence validation works for valid & invalid inputs
4. Output processing filters by confidence and strips heavy polygon points
5. Live smoke test against the serverless endpoint (if ROBOFLOW_API_KEY is set)
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import (
    run_safe_street_workflow,
    _validate_confidence,
    _process_outputs,
    RoboflowAuthError,
    RoboflowValidationError,
    RoboflowWorkflowError,
)

# Public sample image (must be HTTPS)
SAMPLE_IMAGE_URL = "https://media.roboflow.com/notebooks/examples/dog.jpeg"


def test_missing_api_key():
    """Clear error when ROBOFLOW_API_KEY is unset."""
    old_key = os.environ.pop("ROBOFLOW_API_KEY", None)
    try:
        run_safe_street_workflow(SAMPLE_IMAGE_URL)
        assert False, "Expected RoboflowAuthError"
    except RoboflowAuthError:
        print("[PASS] RoboflowAuthError raised on missing API key.")
    finally:
        if old_key is not None:
            os.environ["ROBOFLOW_API_KEY"] = old_key


def test_http_url_rejected():
    """Plain http:// must be rejected."""
    old_key = os.environ.get("ROBOFLOW_API_KEY")
    os.environ["ROBOFLOW_API_KEY"] = "dummy_key"
    try:
        run_safe_street_workflow("http://insecure.com/img.jpg", prefer_sdk=False)
        assert False, "Expected RoboflowValidationError"
    except RoboflowValidationError:
        print("[PASS] Plain http:// URL correctly rejected.")
    finally:
        if old_key is not None:
            os.environ["ROBOFLOW_API_KEY"] = old_key
        else:
            os.environ.pop("ROBOFLOW_API_KEY", None)


def test_confidence_validation():
    """Validate confidence normalization and error handling."""
    assert _validate_confidence(None) is None
    assert _validate_confidence(0.5) == 0.5
    assert _validate_confidence(0.0) == 0.0
    assert _validate_confidence(1.0) == 1.0
    assert _validate_confidence(75) == 0.75  # 75% → 0.75
    assert _validate_confidence(100) == 1.0

    invalid_cases = [-0.1, -10, 101, 150.0, "high", True, False]
    for case in invalid_cases:
        try:
            _validate_confidence(case)
            assert False, f"Expected RoboflowValidationError for: {case}"
        except RoboflowValidationError:
            pass
    print("[PASS] Confidence validation works correctly.")


def test_output_filtering():
    """Verify confidence filtering and polygon stripping."""
    mock_raw = {
        "predictions": {
            "image": {"width": 720, "height": 1280},
            "predictions": [
                {
                    "class": "aqua route Ai",
                    "confidence": 0.91,
                    "x": 323.0,
                    "y": 811.5,
                    "width": 640.0,
                    "height": 911.0,
                    "class_id": 0,
                    "detection_id": "abc",
                    "parent_id": "image",
                    "points": [[10, 20], [30, 40]],
                },
                {
                    "class": "safe street Ai",
                    "confidence": 0.35,
                    "x": 100.0,
                    "y": 200.0,
                    "width": 50.0,
                    "height": 60.0,
                    "class_id": 1,
                    "detection_id": "def",
                    "parent_id": "image",
                },
            ],
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        # No confidence filter — all predictions kept, points stripped
        res_all = _process_outputs(mock_raw, output_dir=tmp_dir)
        preds_all = res_all["predictions"]["predictions"]
        assert len(preds_all) == 2
        assert "points" not in preds_all[0]  # heavy polygon stripped
        print("  [OK] No filter: 2 predictions, points stripped")

        # Filter at 0.5 — low-confidence detection removed
        res_50 = _process_outputs(mock_raw, output_dir=tmp_dir, confidence=0.5)
        preds_50 = res_50["predictions"]["predictions"]
        assert len(preds_50) == 1
        assert preds_50[0]["class"] == "aqua route Ai"
        print("  [OK] conf≥0.5: 1 prediction (aqua route Ai)")

        # Filter at 0.95 — everything filtered out
        res_95 = _process_outputs(mock_raw, output_dir=tmp_dir, confidence=0.95)
        preds_95 = res_95["predictions"]["predictions"]
        assert len(preds_95) == 0
        print("  [OK] conf≥0.95: 0 predictions")

    print("[PASS] Output filtering and stripping works correctly.")


def test_live_smoke():
    """Live test against the real Roboflow serverless endpoint."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "dummy_key":
        print("[SKIP] No valid ROBOFLOW_API_KEY. Skipping live test.")
        return

    print("Running live test with sample image and confidence=0.4 ...")
    with tempfile.TemporaryDirectory() as out_dir:
        result = run_safe_street_workflow(
            image=SAMPLE_IMAGE_URL,
            confidence=0.4,
            output_dir=out_dir,
        )
        assert isinstance(result, dict), "Result must be a dict"
        assert "predictions" in result, (
            f"Expected 'predictions' key in output, got: {list(result.keys())}"
        )

        preds_obj = result["predictions"]
        assert "image" in preds_obj, "predictions must contain 'image' metadata"
        assert "predictions" in preds_obj, "predictions must contain 'predictions' list"

        det_list = preds_obj["predictions"]
        print(f"  Image size: {preds_obj['image']}")
        print(f"  Detections: {len(det_list)}")
        for d in det_list:
            print(
                f"    class={d.get('class')!r}  conf={d.get('confidence'):.3f}  "
                f"box=({d.get('x')}, {d.get('y')}, {d.get('width')}, {d.get('height')})"
            )

        # Assert expected detection keys exist
        if det_list:
            expected_keys = {"x", "y", "width", "height", "confidence", "class", "class_id"}
            actual_keys = set(det_list[0].keys())
            missing = expected_keys - actual_keys
            assert not missing, f"Missing detection keys: {missing}"

    print("[PASS] Live smoke test succeeded!")


if __name__ == "__main__":
    print("=== Offline Tests ===")
    test_missing_api_key()
    test_http_url_rejected()
    test_confidence_validation()
    test_output_filtering()

    print("\n=== Live Smoke Test ===")
    test_live_smoke()
    print("\nAll tests passed!")
