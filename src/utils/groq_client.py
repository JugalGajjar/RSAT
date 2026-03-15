"""
groq_client.py — Groq API client with round-robin key rotation,
retry logic with exponential backoff, and disk-based response caching.

Usage:
    export GROQ_API_KEYS="key1,key2,key3,key4"

    client = GroqRotatingClient()
    result = client.judge_faithfulness("step text", "cell text")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ── Default prompt template ──────────────────────────────────────────────────

FAITHFULNESS_JUDGE_PROMPT = """Evaluate whether the following reasoning step is faithfully supported by the cited table cells.

Reasoning step: "{reasoning_step}"

Cited cell contents: {cited_cells_text}

Judge whether the reasoning step:
1. Is directly supported by the information in the cited cells
2. Does not introduce claims beyond what the cells contain
3. Correctly interprets the cell values (including numerical comparisons)

Respond with ONLY this JSON (no markdown, no extra text):
{{"faithful": true, "score": 0.85, "explanation": "one sentence reason"}}"""

# Load environment variables from .env file if it exists (for local development)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class GroqRotatingClient:
    """
    Groq API client with:
    - Round-robin rotation across multiple API keys
    - Automatic failover on rate-limit errors
    - Disk-based response cache (avoids redundant calls)
    - Exponential backoff via tenacity
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        cache_dir: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        # Gather API keys
        keys_str = os.environ.get("GROQ_API_KEYS", "")
        if keys_str:
            self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single = os.environ.get("GROQ_API_KEY", "")
            if not single:
                raise ValueError(
                    "Set GROQ_API_KEYS (comma-separated) or GROQ_API_KEY env var."
                )
            self.api_keys = [single]

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.current_key_idx = 0

        # Lazily create clients so import doesn't fail if groq isn't installed
        self._clients: list = []

        # Disk cache
        self.cache_dir = Path(cache_dir or ".cache/groq_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, str] = {}

    # ── Client management ────────────────────────────────────────────────────

    @property
    def clients(self) -> list:
        if not self._clients:
            from groq import Groq

            self._clients = [Groq(api_key=k) for k in self.api_keys]
        return self._clients

    def _rotate_key(self) -> None:
        self.current_key_idx = (self.current_key_idx + 1) % len(self.clients)

    # ── Caching ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(messages: list[dict]) -> str:
        blob = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._mem_cache:
            return self._mem_cache[key]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            self._mem_cache[key] = data["response"]
            return data["response"]
        return None

    def _cache_put(self, key: str, response: str) -> None:
        self._mem_cache[key] = response
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({"response": response}))

    # ── API call with rotation ───────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, messages: list[dict]) -> str:
        """Call the Groq API, rotating keys on rate-limit errors."""
        last_error: Optional[Exception] = None

        for _ in range(len(self.clients)):
            client = self.clients[self.current_key_idx]
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as exc:
                err_lower = str(exc).lower()
                if "rate_limit" in err_lower or "429" in err_lower:
                    last_error = exc
                    self._rotate_key()
                    time.sleep(1)
                else:
                    raise

        # All keys hit rate limit — raise so tenacity retries with backoff
        raise last_error  # type: ignore[misc]

    # ── Public interface ─────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Send a message and get a response, with optional caching."""
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        if use_cache:
            key = self._cache_key(messages)
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        response = self._call_api(messages)

        if use_cache:
            self._cache_put(self._cache_key(messages), response)

        return response

    def judge_faithfulness(
        self,
        reasoning_step: str,
        cited_cells_text: str,
        prompt_template: Optional[str] = None,
    ) -> dict:
        """
        Judge whether a reasoning step is faithful to its cited cells.

        Returns:
            {"faithful": bool, "score": float, "explanation": str}
        """
        if not cited_cells_text:
            return {"faithful": False, "score": 0.0, "explanation": "No cells cited."}

        template = prompt_template or FAITHFULNESS_JUDGE_PROMPT
        prompt = template.format(
            reasoning_step=reasoning_step,
            cited_cells_text=cited_cells_text,
        )

        response = self.chat(
            user_message=prompt,
            system_message=(
                "You are a precise faithfulness judge. "
                "Respond ONLY with valid JSON, no markdown."
            ),
        )

        try:
            cleaned = response.strip()
            # Strip optional markdown fences
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

            # Find JSON object
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

            result = json.loads(cleaned)
            return {
                "faithful": bool(result.get("faithful", False)),
                "score": float(result.get("score", 0.0)),
                "explanation": str(result.get("explanation", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return {"faithful": False, "score": 0.0, "explanation": "Parse error."}