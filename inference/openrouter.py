import json
import os
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
    
    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Failed to parse API response: {e}")
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
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
