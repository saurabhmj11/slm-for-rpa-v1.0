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

if /i "%~1"=="install" goto :install
if /i "%~1"=="remove" goto :remove
if /i "%~1"=="check" goto :check

echo [%date% %time%] top-up started >> "%LOG%"
rem training needs three packages the assistant itself does not (quick when already installed)
"%PY%" -m pip install --quiet --disable-pip-version-check wandb psutil pyarrow >> "%LOG%" 2>&1
"%PY%" "%LOCAL%topup_local.py" >> "%LOG%" 2>&1
set "RC=%errorlevel%"
echo [%date% %time%] top-up finished, exit code %RC% >> "%LOG%"
exit /b %RC%

:check
"%PY%" -m pip install --quiet --disable-pip-version-check wandb psutil pyarrow
"%PY%" "%LOCAL%topup_local.py" --check
pause
exit /b 0

:install
schtasks /create /tn "OOA nightly top-up" /tr "\"%~f0\"" /sc daily /st 01:30 /f
rem default task settings skip or stop the run on battery; also start it later if 01:30 was missed
powershell -NoProfile -Command "Set-ScheduledTask -TaskName 'OOA nightly top-up' -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8))" >nul
echo Scheduled: every night at 01:30 (or as soon as the PC is on after a missed time), while you are logged in.
echo The PC is kept awake while a round runs.
pause
exit /b 0

:remove
schtasks /delete /tn "OOA nightly top-up" /f
pause
exit /b 0
