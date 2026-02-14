from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
REQUEST_TIMEOUT_SECONDS = 60

SYSTEM_PROMPT = """Ты — профессиональный редактор новостного Telegram-канала.
Сохраняй ключевые факты, цифры, даты и имена.
Не добавляй ничего от себя.
Не добавляй заголовок и ссылку.
Не включай редакционные или служебные фразы, которые не относятся к сути новости (например призывы написать в бот, рекламу, дисклеймеры).
Пиши кратко, нейтрально, без вводных фраз."""

USER_PROMPT_TEMPLATE = """Сократи текст до не более {limit} символов.
Если превышает — перепиши короче.
Сохрани смысл полностью.

Текст:
---
{text}
---"""

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, limit: int) -> str:
        raise NotImplementedError


class OllamaSummarizer(BaseSummarizer):
    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL) -> None:
        self.model_name = model_name
        self.generate_url = f"{base_url.rstrip('/')}/api/generate"

    def summarize(self, text: str, limit: int) -> str:
        prompt = self._build_prompt(text, limit)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            summarized = (data.get("response") or "").strip()
            if not summarized:
                logging.warning("Ollama model %s returned empty response", self.model_name)
                return truncate_by_sentences(text, limit)
            return enforce_limit(summarized, limit)
        except requests.RequestException as exc:
            logging.warning("Ollama request failed (%s): %s", self.model_name, exc)
            return truncate_by_sentences(text, limit)
        except ValueError as exc:
            logging.warning("Ollama invalid JSON (%s): %s", self.model_name, exc)
            return truncate_by_sentences(text, limit)

    @staticmethod
    def _build_prompt(text: str, limit: int) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(limit=limit, text=text)
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


class SimpleTruncationSummarizer(BaseSummarizer):
    def summarize(self, text: str, limit: int) -> str:
        return truncate_by_sentences(text, limit)


def get_summarizer() -> BaseSummarizer:
    models = _get_available_ollama_models()
    if not models:
        logging.warning("Ollama is unavailable; using simple truncation summarizer")
        return SimpleTruncationSummarizer()

    if _model_available(models, "llama3.1:8b"):
        logging.info("Using Ollama model llama3.1:8b")
        return OllamaSummarizer("llama3.1:8b")

    if _model_available(models, "mistral:7b"):
        logging.info("Using Ollama model mistral:7b")
        return OllamaSummarizer("mistral:7b")

    logging.warning("Preferred models not found in Ollama; fallback to simple truncation")
    return SimpleTruncationSummarizer()


def _get_available_ollama_models() -> list[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        names = [item.get("name", "") for item in models if isinstance(item, dict)]
        return [name for name in names if name]
    except (requests.RequestException, ValueError) as exc:
        logging.warning("Unable to query Ollama models: %s", exc)
        return []


def _model_available(models: list[str], target: str) -> bool:
    normalized = target.lower()
    for model_name in models:
        lower_name = model_name.lower()
        if lower_name == normalized or lower_name.startswith(f"{normalized}:"):
            return True
    return False


def enforce_limit(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return truncate_by_sentences(compact, limit)


def truncate_by_sentences(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact

    sentences = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(compact) if chunk.strip()]
    if sentences:
        selected: list[str] = []
        for sentence in sentences:
            candidate = " ".join(selected + [sentence]).strip()
            if len(candidate) <= limit:
                selected.append(sentence)
            else:
                break
        if selected:
            return " ".join(selected)

    clipped = compact[:limit].rstrip(" ,;:")
    last_space = clipped.rfind(" ")
    if last_space > int(limit * 0.6):
        clipped = clipped[:last_space]
    return clipped.rstrip(" ,;:") + "…"
