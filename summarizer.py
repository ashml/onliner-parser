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
Если в тексте статьи есть слово Белоруссия или Беларусь используй только форму "Беларусь".
Не включай редакционные или служебные фразы, которые не относятся к сути новости (например призывы написать в бот, рекламу, дисклеймеры, ссылки).
Не включай служебные фразы вида 'Перепечатка текста и фотографий Onlíner без разрешения редакции запрещена.', 'Читайте также'.
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


MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "llama": ("llama3.1:8b",),
    "gpt-oss-20b": ("gpt-oss:20b",),
}


def get_supported_model_options() -> tuple[str, ...]:
    return ("auto", *MODEL_ALIASES.keys())


def get_summarizer(model_option: str = "auto", ollama_model: str | None = None) -> BaseSummarizer:
    models = _get_available_ollama_models()
    if not models:
        logging.warning("Ollama is unavailable; using simple truncation summarizer")
        return SimpleTruncationSummarizer()

    if ollama_model:
        selected_model = _find_matching_model_name(models, ollama_model)
        if selected_model:
            logging.info("Using explicitly selected Ollama model %s", selected_model)
            return OllamaSummarizer(selected_model)
        logging.warning("Explicit Ollama model '%s' not found; fallback to configured option", ollama_model)

    selected_model = _resolve_model_name(model_option, models)
    if selected_model:
        logging.info("Using Ollama model %s", selected_model)
        return OllamaSummarizer(selected_model)

    if model_option != "auto":
        logging.warning(
            "Requested model option '%s' not found in Ollama; using simple truncation",
            model_option,
        )
        return SimpleTruncationSummarizer()

    logging.warning("Preferred models not found in Ollama; fallback to simple truncation")
    return SimpleTruncationSummarizer()


def _resolve_model_name(model_option: str, available_models: list[str]) -> str | None:
    if model_option == "auto":
        ordered_candidates = (
            *MODEL_ALIASES["llama"],
            *MODEL_ALIASES["gpt-oss-20b"],
            "mistral:7b",
        )
    else:
        ordered_candidates = MODEL_ALIASES.get(model_option, ())

    for candidate in ordered_candidates:
        resolved = _find_matching_model_name(available_models, candidate)
        if resolved:
            return resolved
    return None


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


def _find_matching_model_name(models: list[str], target: str) -> str | None:
    normalized = target.lower()
    for model_name in models:
        lower_name = model_name.lower()
        if lower_name == normalized or lower_name.startswith(f"{normalized}:"):
            return model_name
    return None


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
