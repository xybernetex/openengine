"""
llm_client.py — Xybernetex Multi-Provider LLM Abstraction

Normalises five providers to a single .chat() call:
    cloudflare  — CF Workers AI (default)
    openai      — OpenAI Chat Completions
    mistral     — Mistral AI (OpenAI-compatible endpoint)
    anthropic   — Anthropic Messages API
    gemini      — Google Gemini generateContent API

Environment variables:
    LLM_PROVIDER   cloudflare | openai | mistral | anthropic | gemini
    LLM_MODEL      optional model override (sensible default per provider)
    LLM_API_KEY    API key (not needed for cloudflare — uses CF_ACCOUNT_ID / CF_API_TOKEN)

Cloudflare-specific (only when LLM_PROVIDER=cloudflare):
    CF_ACCOUNT_ID
    CF_API_TOKEN
"""
from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

# ── Provider defaults ──────────────────────────────────────────────────────────
_DEFAULT_MODELS: dict[str, str] = {
    "cloudflare": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "openai":     "gpt-4o",
    "mistral":    "mistral-large-latest",
    "anthropic":  "claude-opus-4-6",
    "gemini":     "gemini-2.0-flash",
}

_OPENAI_COMPAT_BASE: dict[str, str] = {
    "openai":  "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}


class LLMClient:
    """
    Single interface for all supported LLM providers.

    Usage:
        client = LLMClient()
        text   = client.chat(messages, max_tokens=1024, temperature=0.5)
    """

    def __init__(
        self,
        provider   : str = "",
        api_key    : str = "",
        model      : str = "",
        cf_account : str = "",
        cf_token   : str = "",
    ) -> None:
        self.provider = (provider.lower().strip() if provider else "") or os.getenv("LLM_PROVIDER", "cloudflare").lower().strip()
        self.api_key  = api_key.strip() if api_key else os.getenv("LLM_API_KEY", "").strip()

        default_model = _DEFAULT_MODELS.get(self.provider, _DEFAULT_MODELS["cloudflare"])
        self.model    = (model.strip() if model else "") or os.getenv("LLM_MODEL", default_model).strip() or default_model

        # Cloudflare-specific
        self._cf_account = (cf_account.strip() if cf_account else "") or os.getenv("CF_ACCOUNT_ID", "").strip()
        self._cf_token   = (cf_token.strip()   if cf_token   else "") or os.getenv("CF_API_TOKEN",  "").strip()

        if self.provider not in _DEFAULT_MODELS:
            logger.warning(
                "Unknown LLM_PROVIDER=%r — falling back to cloudflare. "
                "Valid options: %s",
                self.provider,
                ", ".join(_DEFAULT_MODELS),
            )
            self.provider = "cloudflare"

        logger.info("LLMClient: provider=%s model=%s", self.provider, self.model)

    # ── Public interface ───────────────────────────────────────────────────────

    def chat(
        self,
        messages   : list[dict[str, str]],
        max_tokens : int   = 2048,
        temperature: float = 0.5,
    ) -> str:
        """
        Send a chat request and return the response text.

        Never raises — returns a descriptive error token on failure so callers
        can handle degraded responses gracefully.
        """
        try:
            if self.provider == "cloudflare":
                return self._call_cloudflare(messages, max_tokens, temperature)
            if self.provider in ("openai", "mistral"):
                return self._call_openai_compat(messages, max_tokens, temperature)
            if self.provider == "anthropic":
                return self._call_anthropic(messages, max_tokens, temperature)
            if self.provider == "gemini":
                return self._call_gemini(messages, max_tokens, temperature)
        except Exception as exc:  # noqa: BLE001
            return f"[LLM_ERROR: {exc}]"
        return "[LLM_ERROR: unknown provider]"

    # ── Cloudflare Workers AI ──────────────────────────────────────────────────

    def _call_cloudflare(
        self,
        messages   : list[dict[str, str]],
        max_tokens : int,
        temperature: float,
    ) -> str:
        if not self._cf_account or not self._cf_token:
            return "[LLM_ERROR: CF_ACCOUNT_ID or CF_API_TOKEN not set]"

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._cf_account}/ai/run/{self.model}"
        )
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self._cf_token}"},
            json={
                "messages"   : messages,
                "max_tokens" : max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result   = resp.json().get("result", {})
        response = result.get("response", "")
        if isinstance(response, (dict, list)):
            response = json.dumps(response)
        return str(response).strip()

    # ── OpenAI / Mistral (OpenAI-compatible) ──────────────────────────────────

    def _call_openai_compat(
        self,
        messages   : list[dict[str, str]],
        max_tokens : int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            return f"[LLM_ERROR: LLM_API_KEY not set for provider={self.provider}]"

        base = _OPENAI_COMPAT_BASE[self.provider]
        resp = requests.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       self.model,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Anthropic Messages API ─────────────────────────────────────────────────

    def _call_anthropic(
        self,
        messages   : list[dict[str, str]],
        max_tokens : int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            return "[LLM_ERROR: LLM_API_KEY not set for provider=anthropic]"

        # Anthropic separates the system prompt from the messages array
        system  = ""
        payload = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                payload.append({"role": m["role"], "content": m["content"]})

        body: dict[str, object] = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    payload,
        }
        if system:
            body["system"] = system

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type":      "application/json",
            },
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    # ── Google Gemini ──────────────────────────────────────────────────────────

    def _call_gemini(
        self,
        messages   : list[dict[str, str]],
        max_tokens : int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            return "[LLM_ERROR: LLM_API_KEY not set for provider=gemini]"

        # Gemini uses "user"/"model" roles and a different message structure.
        # System prompt goes in a top-level systemInstruction field.
        system_text = ""
        contents    = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
                continue
            gemini_role = "model" if m["role"] == "assistant" else "user"
            contents.append({
                "role":  gemini_role,
                "parts": [{"text": m["content"]}],
            })

        body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature":     temperature,
            },
        }
        if system_text:
            body["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return (
            resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
