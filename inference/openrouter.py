import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables (try .env.local first, then .env)
load_dotenv('.env.local')
load_dotenv('.env')

# Support both OPENROUTER_API_KEY and OPENROUTER_KEY
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    model: str,
    messages: list[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Optional[str]:
    """Call OpenRouter API and return the response content."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    max_retries = 3
    backoffs = [1, 2, 4]  # seconds, one per retry attempt

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not content:
                # Empty content is scored as api_failure upstream; log why so it is
                # diagnosable (e.g. GLM 5.3 returned reasoning-only responses in
                # run-2026-09 with no error line in the log). Retry once like a 5xx:
                # reasoning-model providers sometimes return empty content transiently.
                usage = data.get("usage") or {}
                print(
                    f"Empty content from {model} (finish_reason={choice.get('finish_reason')}, "
                    f"native_finish={choice.get('native_finish_reason')}, usage={usage}, "
                    f"attempt {attempt + 1}/{max_retries + 1})",
                    file=sys.stderr,
                )
                if attempt < max_retries:
                    time.sleep(backoffs[attempt])
                    continue
                return None
            return content

        except requests.exceptions.RequestException as e:
            status = e.response.status_code if e.response is not None else None
            # Retry on transient failures: connection/timeout errors (no response
            # at all) and HTTP 429 / 5xx. Fail fast on other 4xx (invalid model
            # ID, ZDR policy blocks, etc.) - retrying those just wastes time.
            retryable = status is None or status == 429 or (status is not None and 500 <= status < 600)

            # Log the response body because OpenRouter error details (invalid model
            # ID, ZDR policy blocks, provider capacity) only appear there.
            body = ""
            if e.response is not None:
                try:
                    body = e.response.text[:500]
                except Exception:
                    body = "<unreadable body>"

            if retryable and attempt < max_retries:
                delay = backoffs[attempt]
                print(
                    f"API request failed (status={status}, attempt {attempt + 1}/{max_retries + 1}): "
                    f"{e} | body: {body} | retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue

            print(f"API request failed: {e} | body: {body}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Failed to parse API response: {e}")
            return None

    return None


def load_cases(path):
    """Load cases from file. Handles both plain list and metadata format."""
    with open(path) as f:
        data = json.load(f)
    
    # Handle new format with metadata
    if isinstance(data, dict) and "cases" in data:
        return data["cases"], data.get("metadata")
    
    # Handle old format (plain list)
    return data, None


def write_predictions(path, predictions, metadata=None):
    """Write predictions with optional metadata."""
    output = predictions
    
    # If metadata provided, wrap predictions with it
    if metadata:
        output = {
            "metadata": metadata,
            "predictions": predictions
        }
    
    # Atomic write: write to a temp file in the same directory, then rename over
    # the target. A crash mid-write leaves the previous (or no) file intact
    # rather than a truncated/corrupt one - important now that this is called
    # repeatedly as a checkpoint, not just once at the end.
    path = str(path)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, path)
