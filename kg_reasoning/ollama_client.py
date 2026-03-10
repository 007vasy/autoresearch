"""Thin HTTP wrapper for the ollama generate API."""

import time
import logging

import httpx

import config

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def generate(
    prompt: str,
    model: str = config.OLLAMA_MODEL,
    temperature: float = 0.0,
    timeout: int = config.OLLAMA_TIMEOUT,
) -> str:
    """Call ollama /api/generate and return the response text."""
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["response"]
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            log.warning("ollama attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def check_available() -> bool:
    """Return True if ollama is reachable and the model is loaded."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{config.OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return config.OLLAMA_MODEL in models
    except Exception:
        return False
