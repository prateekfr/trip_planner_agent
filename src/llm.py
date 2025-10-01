import os
import json
import time
from typing import Dict, Any, List, Optional
import requests

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterError(RuntimeError):
    pass

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY not set. Put it in .env")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        http_ref = os.getenv("HTTP_REFERER")
        x_title = os.getenv("X_TITLE")
        if http_ref:
            headers["HTTP-Referer"] = http_ref
        if x_title:
            headers["X-Title"] = x_title
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(BASE_URL, headers=self._headers(), json=payload, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(0.75 * (attempt + 1))
        raise OpenRouterError(f"OpenRouter request failed: {last_err}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> Any:
        """Ask model to reply with strict JSON; attempts to parse and re-ask once if needed."""
        sys_msg = {
            "role": "system",
            "content": (
                "Respond with STRICT JSON only. Do not include prose or code fences. "
                "If you include strings, ensure valid JSON quoting."
            ),
        }
        combined = [sys_msg] + messages
        text = self.chat(combined, model=model, temperature=temperature, timeout=timeout, max_retries=max_retries)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fix_prompt = [
                sys_msg,
                {"role": "user", "content": f"The previous output was not valid JSON. Fix and return ONLY valid JSON:\n{text}"}
            ]
            fixed = self.chat(fix_prompt, model=model, temperature=0.0, timeout=timeout, max_retries=max_retries)
            return json.loads(fixed)
