@echo off
rem ====================================================================================
rem  OOA overnight top-up (Windows side). Runs training\local\topup_local.py inside WSL.
rem    topup_nightly.bat            run one round now (about 2 hours on a 4-core CPU)
rem    topup_nightly.bat install    run it every night at 01:30 (Windows Task Scheduler)
rem    topup_nightly.bat remove     stop the nightly runs
rem  Needs the one-time WSL setup first (README.md in this folder, "Overnight top-up").
rem  Output goes to topup_nightly.log next to this file.
rem ====================================================================================
if /i "%~1"=="install" (
  schtasks /create /tn "OOA nightly top-up" /tr "\"%~f0\"" /sc daily /st 01:30 /f
  echo Scheduled: every night at 01:30, while you are logged in and the PC is on ^(not asleep^).
  pause & exit /b 0
)
if /i "%~1"=="remove" (
  schtasks /delete /tn "OOA nightly top-up" /f
  pause & exit /b 0
)
echo [%date% %time%] top-up started >> "%~dp0topup_nightly.log"
wsl --cd "%~dp0." -e bash -lc "~/ooa-train-venv/bin/python topup_local.py" >> "%~dp0topup_nightly.log" 2>&1
echo [%date% %time%] top-up finished, exit code %errorlevel% >> "%~dp0topup_nightly.log"
