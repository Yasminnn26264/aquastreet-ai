import * as fs from 'fs';
import * as path from 'path';

// ── Configuration ──────────────────────────────────────────────────────
const WORKSPACE_NAME = "yasmins-workspace-n6tkr";
const WORKFLOW_ID =
  "safe-street-ai-1-rfdetr-small-t1-vsafe-street-ai-1-rfdetr-small-t1-logic";
const ENDPOINT = `https://serverless.roboflow.com/${WORKSPACE_NAME}/workflows/${WORKFLOW_ID}`;

// ── Types ──────────────────────────────────────────────────────────────
/** A single detection box returned by the workflow. */
export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class_id: number;
  class: string;
  detection_id: string;
  parent_id: string;
}

/** The predictions payload (matches the real API response shape). */
export interface PredictionsOutput {
  image: { width: number; height: number };
  predictions: Detection[];
}

/** Options for running the workflow. */
export interface WorkflowOptions {
  /** Confidence threshold: 0.0–1.0 or 0–100%. */
  confidence?: number;
  /** Additional workflow parameters (iou_threshold, max_detections, etc.). */
  parameters?: Record<string, any>;
  /** Directory where base64 image outputs are written. */
  outputDir?: string;
  /** Request timeout in milliseconds (default 30 000). */
  timeoutMs?: number;
  /** Maximum retry attempts on transient errors (default 3). */
  maxRetries?: number;
}

// ── Errors ─────────────────────────────────────────────────────────────
export class RoboflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RoboflowError";
  }
}
export class RoboflowAuthError extends RoboflowError {
  constructor(message: string) {
    super(message);
    this.name = "RoboflowAuthError";
  }
}
export class RoboflowValidationError extends RoboflowError {
  constructor(message: string) {
    super(message);
    this.name = "RoboflowValidationError";
  }
}

// ── Helpers ────────────────────────────────────────────────────────────
function validateConfidence(confidence?: number): number | undefined {
  if (confidence === undefined || confidence === null) return undefined;
  if (typeof confidence !== "number" || isNaN(confidence) || confidence < 0) {
    throw new RoboflowValidationError(
      `Invalid confidence value: ${confidence}. Must be [0.0, 1.0] or percentage [0, 100].`
    );
  }
  if (confidence > 100.0) {
    throw new RoboflowValidationError(
      `Confidence cannot exceed 1.0 (or 100%). Got: ${confidence}`
    );
  }
  return confidence > 1.0 ? confidence / 100.0 : confidence;
}

function filterAndClean(item: any, minConf?: number): any {
  if (Array.isArray(item)) {
    const out: any[] = [];
    for (const el of item) {
      const c = filterAndClean(el, minConf);
      if (c !== null && c !== undefined) out.push(c);
    }
    return out;
  }
  if (item && typeof item === "object") {
    const conf = item.confidence ?? item.score;
    if (minConf !== undefined && typeof conf === "number" && conf < minConf) {
      return null;
    }
    const cleaned: Record<string, any> = {};
    for (const [k, v] of Object.entries(item)) {
      if (k === "points") continue; // strip heavy polygon coords
      const cv = filterAndClean(v, minConf);
      if (cv !== null && cv !== undefined) cleaned[k] = cv;
    }
    return cleaned;
  }
  return item;
}

// ── Main function ──────────────────────────────────────────────────────
/**
 * Run the Safe Street AI object-detection workflow.
 *
 * @param imageInput - Local file path or HTTPS URL of the image to analyze.
 * @param options - See {@link WorkflowOptions}.
 * @returns A dict keyed by output names (currently `predictions`).
 */
export async function runSafeStreetWorkflow(
  imageInput: string,
  options: WorkflowOptions = {}
): Promise<Record<string, any>> {
  const apiKey = process.env.ROBOFLOW_API_KEY;
  if (!apiKey) {
    throw new RoboflowAuthError(
      "ROBOFLOW_API_KEY environment variable is not set. " +
        "Get it at https://app.roboflow.com/settings/api"
    );
  }

  const {
    confidence,
    parameters,
    outputDir = "./output_images",
    timeoutMs = 30_000,
    maxRetries = 3,
  } = options;

  const confVal = validateConfidence(confidence);

  // ── Build image payload ──
  let imagePayload: { type: string; value: string };
  if (imageInput.startsWith("https://")) {
    imagePayload = { type: "url", value: imageInput };
  } else if (imageInput.startsWith("http://")) {
    throw new RoboflowValidationError(
      "Image URL must use HTTPS (plain HTTP is rejected by Roboflow)."
    );
  } else {
    const resolved = path.resolve(imageInput);
    if (!fs.existsSync(resolved)) {
      throw new RoboflowError(`Image file not found: ${resolved}`);
    }
    const buf = fs.readFileSync(resolved);
    imagePayload = { type: "base64", value: buf.toString("base64") };
  }

  // ── Build request body ──
  const params: Record<string, any> = { ...(parameters || {}) };
  if (confVal !== undefined) {
    params.confidence ??= confVal;
    params.confidence_threshold ??= confVal;
  }
  const body: Record<string, any> = { inputs: { image: imagePayload } };
  if (Object.keys(params).length > 0) body.parameters = params;

  // ── Retry loop ──
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), timeoutMs);

      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      clearTimeout(timer);

      if (res.status === 401 || res.status === 403) {
        throw new RoboflowAuthError(
          `Authentication failed (HTTP ${res.status})`
        );
      }
      if (!res.ok) {
        throw new RoboflowError(
          `Roboflow API error (${res.status}): ${await res.text()}`
        );
      }

      const json: any = await res.json();

      // REST returns {"outputs": [{...}], "profiler_trace": ...}
      let rawResult: Record<string, any>;
      if (json && typeof json === "object" && Array.isArray(json.outputs)) {
        rawResult = json.outputs[0] ?? {};
      } else if (Array.isArray(json)) {
        rawResult = json[0] ?? {};
      } else {
        rawResult = json;
      }

      // ── Process outputs ──
      fs.mkdirSync(outputDir, { recursive: true });
      const processed: Record<string, any> = {};

      for (const [key, val] of Object.entries(rawResult)) {
        // base64 image string
        if (
          typeof val === "string" &&
          (val.startsWith("data:image/") ||
            (val.length > 1000 && /^[A-Za-z0-9+/=]+$/.test(val)))
        ) {
          const b64 = val.includes(",") ? val.split(",")[1] : val;
          const outPath = path.join(outputDir, `output_${key}.jpg`);
          fs.writeFileSync(outPath, Buffer.from(b64, "base64"));
          processed[key] = { type: "saved_image", path: outPath };
          continue;
        }
        // dict with explicit base64
        if (
          val &&
          typeof val === "object" &&
          (val as any).type === "base64" &&
          (val as any).value
        ) {
          const b64 = (val as any).value.includes(",")
            ? (val as any).value.split(",")[1]
            : (val as any).value;
          const outPath = path.join(outputDir, `output_${key}.jpg`);
          fs.writeFileSync(outPath, Buffer.from(b64, "base64"));
          processed[key] = { type: "saved_image", path: outPath };
          continue;
        }
        processed[key] = filterAndClean(val, confVal);
      }

      return processed;
    } catch (err: any) {
      lastError = err;
      if (err instanceof RoboflowAuthError) throw err;
      if (attempt < maxRetries) {
        const wait = Math.pow(1.5, attempt) * 1000;
        await new Promise((r) => setTimeout(r, wait));
      }
    }
  }

  throw new RoboflowError(
    `Workflow failed after ${maxRetries} attempts: ${lastError?.message}`
  );
}
