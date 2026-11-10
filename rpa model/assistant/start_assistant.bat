@echo off
rem ===================== OOA Assistant launcher (Windows) =====================
rem Edit the paths below to match where you put things (see README.md, "Setup").
rem No quotes inside the values; spaces in folder names are fine.

set PYTHON=python
set MODEL_DIR=C:\ooa\nanochat_base
set NANOCHAT_DIR=C:\ooa\nanochat
set PLATFORM_DIR=C:\ooa\oneofficeautomation-platform
set CATALOG=C:\ooa\sft\catalog.jsonl
set DATA_DIR=%~dp0assistant_data
set PORT=8765

rem Set to 1 to try the app without the real model (canned answers only).
set FAKE_MODEL=0
rem ============================================================================

set EXTRA=
if "%FAKE_MODEL%"=="1" set EXTRA=--fake-model

if not exist "%PLATFORM_DIR%\contracts\flow-schema\flow.v1.json" (
  echo Cannot find the platform repo at "%PLATFORM_DIR%". Edit PLATFORM_DIR at the top of this file.
  pause
  exit /b 1
)
if not exist "%CATALOG%" (
  echo Cannot find the catalog at "%CATALOG%". Edit CATALOG at the top of this file.
  pause
  exit /b 1
)
if "%FAKE_MODEL%"=="0" if not exist "%MODEL_DIR%\chatsft_checkpoints\d8" (
  echo Cannot find the model at "%MODEL_DIR%\chatsft_checkpoints\d8". Edit MODEL_DIR at the top of this file.
  pause
  exit /b 1
)

echo Starting the OOA Assistant. Your browser will open; if the page is not ready yet, wait for
echo "OOA Assistant is running" below and refresh. Close this window to stop it.
start "" "http://127.0.0.1:%PORT%/"
"%PYTHON%" "%~dp0app.py" --model-dir "%MODEL_DIR%" --nanochat "%NANOCHAT_DIR%" --platform "%PLATFORM_DIR%" --catalog "%CATALOG%" --data-dir "%DATA_DIR%" --port %PORT% %EXTRA%
pause
