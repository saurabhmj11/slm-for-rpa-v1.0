@echo off
rem ====================================================================================
rem  OOA overnight top-up, directly on Windows (no WSL, no graphics card needed).
rem    topup_nightly.bat check      first time: check that everything is in place (shows on screen)
rem    topup_nightly.bat            run one round now (about 2 hours on a 4-core CPU); output in topup_nightly.log
rem    topup_nightly.bat install    run it every night at 01:30 (Windows Task Scheduler)
rem    topup_nightly.bat remove     stop the nightly runs
rem  Uses the same Python as the assistant (the "set PYTHON=" line of assistant\start_assistant.bat).
rem ====================================================================================
setlocal
set "LOCAL=%~dp0"
set "LOG=%~dp0topup_nightly.log"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"set PYTHON=" "%LOCAL%..\..\assistant\start_assistant.bat"') do set "PY=%%b"
if not defined PY set "PY=python"

if /i "%~1"=="install" (
  schtasks /create /tn "OOA nightly top-up" /tr "\"%~f0\"" /sc daily /st 01:30 /f
  echo Scheduled: every night at 01:30, while you are logged in and the PC is on ^(not asleep^).
  pause & exit /b 0
)
if /i "%~1"=="remove" (
  schtasks /delete /tn "OOA nightly top-up" /f
  pause & exit /b 0
)

rem training needs three packages the assistant itself does not (quick when already installed)
"%PY%" -m pip install --quiet --disable-pip-version-check wandb psutil pyarrow

if /i "%~1"=="check" (
  "%PY%" "%LOCAL%topup_local.py" --check
  pause & exit /b 0
)
echo [%date% %time%] top-up started >> "%LOG%"
"%PY%" "%LOCAL%topup_local.py" >> "%LOG%" 2>&1
echo [%date% %time%] top-up finished, exit code %errorlevel% >> "%LOG%"
