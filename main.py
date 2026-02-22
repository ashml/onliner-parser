#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from onliner_bot import run_app


SETTINGS_FILE = Path("launcher_settings.json")
DEFAULT_MODEL = "llama"
MODEL_ALIASES = {"auto", "llama", "gpt-oss-20b"}


def load_launcher_settings() -> dict[str, str]:
    if not SETTINGS_FILE.exists():
        return {"model": DEFAULT_MODEL}

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"model": DEFAULT_MODEL}

    model = settings.get("model")
    if not isinstance(model, str) or not model.strip():
        return {"model": DEFAULT_MODEL}
    return {"model": model.strip()}


def save_launcher_settings(settings: dict[str, str]) -> None:
    with SETTINGS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)


def ask_choice(prompt: str, allowed: set[str]) -> str:
    while True:
        value = input(prompt).strip()
        if value in allowed:
            return value
        print(f"Неверный выбор. Допустимые варианты: {', '.join(sorted(allowed))}")


def ask_model() -> tuple[str, str | None]:
    print("\n=== Меню выбора модели ===")
    print("1 - auto")
    print("2 - llama")
    print("3 - gpt-oss-20b")
    print("4 - ввести точное имя модели из ollama list")
    print("0 - назад в основное меню")

    selected = ask_choice("Выберите модель (0-4): ", {"0", "1", "2", "3", "4"})
    if selected == "0":
        return "back", None
    if selected == "1":
        return "auto", None
    if selected == "2":
        return "llama", None
    if selected == "3":
        return "gpt-oss-20b", None

    while True:
        custom = input("Введите точное имя модели: ").strip()
        if custom:
            return "auto", custom
        print("Имя модели не может быть пустым.")


def main() -> None:
    load_dotenv()
    settings = load_launcher_settings()
    selected_model = settings.get("model", DEFAULT_MODEL)

    while True:
        print("\n=== Основное меню ===")
        print("Режим работы:")
        print("1 - last24h")
        print("2 - watch")
        print("\nВыбор модели:")
        print(f"Текущая модель: {selected_model}")
        print("0 - открыть меню выбора модели")

        selected = ask_choice("Выберите пункт (0-2): ", {"0", "1", "2"})
        if selected == "0":
            model, custom_model = ask_model()
            if model == "back":
                continue
            selected_model = custom_model if custom_model else model
            settings["model"] = selected_model
            save_launcher_settings(settings)
            print(f"Выбрана модель: {selected_model}")
            continue

        mode = "last24h" if selected == "1" else "watch"
        break

    model = selected_model if selected_model in MODEL_ALIASES else "auto"
    ollama_model = None if selected_model in MODEL_ALIASES else selected_model

    args = argparse.Namespace(
        mode=mode,
        interval=300,
        state_file="state.json",
        log_level="INFO",
        model=model,
        ollama_model=ollama_model,
    )
    run_app(args)


if __name__ == "__main__":
    main()
