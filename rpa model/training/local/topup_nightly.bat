@echo off
rem ====================================================================================
rem  OOA nightly top-up, directly on Windows (no WSL, no graphics card needed).
rem    topup_nightly.bat check      first time: check that everything is in place (shows on screen)
rem    topup_nightly.bat            run now: continue the top-up run (saved every 50 steps), and when it is
rem                                 finished test it and install it only if it is better; output in topup_nightly.log
rem    topup_nightly.bat install    run it every night at 01:30 (Windows Task Scheduler, at most 8 hours a night)
rem    topup_nightly.bat remove     stop the nightly runs
rem  Needs the data and model that ooa_local_pc.ipynb (cells 1-3) downloaded into "rpa model\train_data".
rem  Uses the same Python as the assistant (the "set PYTHON=" line of assistant\start_assistant.bat).
rem ====================================================================================
setlocal
set "LOCAL=%~dp0"
set "RPA=%~dp0..\.."
set "DATA=%RPA%\train_data\sft_render"
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
"%PY%" "%LOCAL%data_pc.py" --rpa "%RPA%" --feedback-only >> "%LOG%" 2>&1
"%PY%" "%LOCAL%train_pc.py" --rpa "%RPA%" --mode topup --data "%DATA%" >> "%LOG%" 2>&1
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :done
"%PY%" "%LOCAL%eval_pc.py" --rpa "%RPA%" --run-state "%RPA%\local_run\topup\state.json" --install >> "%LOG%" 2>&1
set "RC=%errorlevel%"
:done
echo [%date% %time%] top-up finished, exit code %RC% >> "%LOG%"
exit /b %RC%

:check
"%PY%" "%LOCAL%train_pc.py" --rpa "%RPA%" --mode topup --data "%DATA%" --check
pause
exit /b 0

:install
schtasks /create /tn "OOA nightly top-up" /tr "\"%~f0\"" /sc daily /st 01:30 /f
rem default task settings skip or stop the run on battery; also start it later if 01:30 was missed
powershell -NoProfile -Command "Set-ScheduledTask -TaskName 'OOA nightly top-up' -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8))" >nul
echo Scheduled: every night at 01:30 (or as soon as the PC is on after a missed time), while you are logged in.
echo The PC is kept awake while a round runs; a round that does not finish in one night continues the next night.
pause
exit /b 0

:remove
schtasks /delete /tn "OOA nightly top-up" /f
pause
exit /b 0
