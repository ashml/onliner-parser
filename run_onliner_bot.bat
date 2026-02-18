@echo off
chcp 65001 >nul
cd /d %~dp0

echo ==============================
echo   Onliner Tech Telegram Bot
echo ==============================
echo.
echo Modes:
echo 1 - last24h
echo 2 - watch
set /p mode=Choose mode (1 or 2): 

echo.
echo Available Ollama models:
ollama list
echo.
echo Summarizer model options:
echo 1 - auto (llama -^> gpt-oss-20b -^> mistral)
echo 2 - llama
echo 3 - gpt-oss-20b
echo 4 - exact model name from ollama list
set /p model_mode=Choose model option (1-4): 

set "MODEL_ARGS="
if "%model_mode%"=="1" set "MODEL_ARGS=--model auto"
if "%model_mode%"=="2" set "MODEL_ARGS=--model llama"
if "%model_mode%"=="3" set "MODEL_ARGS=--model gpt-oss-20b"
if "%model_mode%"=="4" (
    set /p custom_model=Enter exact model name: 
    set "MODEL_ARGS=--model auto --ollama-model %custom_model%"
)

if "%MODEL_ARGS%"=="" (
    echo Invalid model choice
    pause
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate

if "%mode%"=="1" (
    python onliner_bot.py last24h %MODEL_ARGS%
) else if "%mode%"=="2" (
    python onliner_bot.py watch --interval 300 %MODEL_ARGS%
) else (
    echo Invalid mode choice
)

pause
