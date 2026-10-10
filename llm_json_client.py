#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_json_client.py — Resilient multi-provider LLM client that returns parsed JSON.

Why this exists:
  The carousel pipeline used to call OpenRouter + a hardcoded list of Gemini models.
  In CI the Gemini free-tier quota was exhausted (429), the gemini-1.5-* models were
  retired (404), and OpenRouter failures were swallowed silently. Every provider failed,
  so the pipeline fell back to a generic hardcoded dialogue template — producing the
  same "What makes this happen in modern engineering?" slides on every run.

Provider chain (each is tried in order, every failure is logged):
  1. Gemini (several models — each has its own free-tier quota bucket)
  2. OpenRouter (several models, per-model error isolation)
  3. Cloudflare Workers AI (CF_API_TOKEN + CF_ACCOUNT_ID)
  4. Hugging Face Inference Router (HF_TOKEN)
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
]
CLOUDFLARE_MODELS = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct"]
HF_MODELS = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]


def extract_json(raw: Any) -> Optional[Any]:
    """Parse JSON from an LLM response, tolerating code fences and surrounding prose."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _short(err: Any) -> str:
    return str(err).replace("\n", " ")[:160]


def _try_gemini(prompt: str, temperature: float, label: str) -> Optional[Dict]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai  # google-genai SDK
    except Exception:
        genai = None
    for model in GEMINI_MODELS:
        try:
            if genai is not None:
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": temperature},
                )
                raw = resp.text
            else:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                raw = genai_legacy.GenerativeModel(model).generate_content(prompt).text
            data = extract_json(raw)
            if isinstance(data, dict):
                print(f"✅ [{label}] Gemini ({model}) responded")
                return data
            print(f"⚠️ [{label}] Gemini ({model}) returned non-JSON output")
        except Exception as e:
            print(f"⚠️ [{label}] Gemini ({model}) failed: {_short(e)}")
    return None


def _try_openrouter(prompt: str, temperature: float, label: str) -> Optional[Dict]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vjaab/YtDidYouKnowByVJ",
        "X-Title": "YtDidYouKnowByVJ",
    }
    for model in OPENROUTER_MODELS:
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            if res.status_code != 200:
                print(f"⚠️ [{label}] OpenRouter ({model}) HTTP {res.status_code}: {_short(res.text)}")
                continue
            raw = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            data = extract_json(raw)
            if isinstance(data, dict):
                print(f"✅ [{label}] OpenRouter ({model}) responded")
                return data
            print(f"⚠️ [{label}] OpenRouter ({model}) returned non-JSON output")
        except Exception as e:
            print(f"⚠️ [{label}] OpenRouter ({model}) failed: {_short(e)}")
    return None


def _try_cloudflare(prompt: str, temperature: float, label: str) -> Optional[Dict]:
    token = os.getenv("CF_API_TOKEN", "")
    account = os.getenv("CF_ACCOUNT_ID", "")
    if not token or not account:
        return None
    for model in CLOUDFLARE_MODELS:
        try:
            res = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "messages": [
                        {"role": "system", "content": "You reply with valid JSON only. No markdown, no prose."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2500,
                    "temperature": temperature,
                },
                timeout=60,
            )
            if res.status_code != 200:
                print(f"⚠️ [{label}] Cloudflare ({model}) HTTP {res.status_code}: {_short(res.text)}")
                continue
            raw = (res.json().get("result") or {}).get("response", "")
            data = extract_json(raw)
            if isinstance(data, dict):
                print(f"✅ [{label}] Cloudflare Workers AI ({model}) responded")
                return data
            print(f"⚠️ [{label}] Cloudflare ({model}) returned non-JSON output")
        except Exception as e:
            print(f"⚠️ [{label}] Cloudflare ({model}) failed: {_short(e)}")
    return None


def _try_huggingface(prompt: str, temperature: float, label: str) -> Optional[Dict]:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        return None
    for model in HF_MODELS:
        try:
            res = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2500,
                },
                timeout=60,
            )
            if res.status_code != 200:
                print(f"⚠️ [{label}] HuggingFace ({model}) HTTP {res.status_code}: {_short(res.text)}")
                continue
            raw = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            data = extract_json(raw)
            if isinstance(data, dict):
                print(f"✅ [{label}] HuggingFace ({model}) responded")
                return data
            print(f"⚠️ [{label}] HuggingFace ({model}) returned non-JSON output")
        except Exception as e:
            print(f"⚠️ [{label}] HuggingFace ({model}) failed: {_short(e)}")
    return None


def query_json(prompt: str, temperature: float = 0.4, label: str = "LLM", validator=None) -> Optional[Dict]:
    """
    Query every configured provider in turn and return the first JSON dict that passes
    `validator(data) -> bool` (if given). Returns None only when ALL providers fail.
    """
    for provider in (_try_gemini, _try_openrouter, _try_cloudflare, _try_huggingface):
        data = provider(prompt, temperature, label)
        if data is None:
            continue
        if validator is None or validator(data):
            return data
        print(f"⚠️ [{label}] {provider.__name__[5:]} response failed validation, trying next provider...")
    print(f"❌ [{label}] All LLM providers failed")
    return None
