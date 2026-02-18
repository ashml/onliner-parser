#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dotenv import load_dotenv

from onliner_bot import run_app


def ask_choice(prompt: str, allowed: set[str]) -> str:
    while True:
        value = input(prompt).strip()
        if value in allowed:
            return value
        print(f"Неверный выбор. Допустимые варианты: {', '.join(sorted(allowed))}")


def ask_mode() -> str:
    print("Режимы:")
    print("1 - last24h")
    print("2 - watch")
    selected = ask_choice("Выберите режим (1 или 2): ", {"1", "2"})
    return "last24h" if selected == "1" else "watch"


def ask_model() -> tuple[str, str | None]:
    print("\nДоступные варианты модели:")
    print("1 - auto")
    print("2 - llama")
    print("3 - gpt-oss-20b")
    print("4 - ввести точное имя модели из ollama list")

    selected = ask_choice("Выберите модель (1-4): ", {"1", "2", "3", "4"})
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

    mode = ask_mode()
    model, ollama_model = ask_model()

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
