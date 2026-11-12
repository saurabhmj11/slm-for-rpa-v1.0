@echo off
rem ===================== ONEOFFICEAUTOMATION platform + OOA Assistant (Windows) =====================
rem Starts the platform (One Command Center, One Agent Studio, a One Runner) from its source code, and this
rem assistant next to it, each in its own window. First run: downloads the platform and builds it (10-20 minutes).
rem Needs Git and a Java 21 JDK. Edit PLATFORM_SRC to keep the platform somewhere else.
setlocal
set PLATFORM_SRC=%USERPROFILE%\OOA\oneofficeautomation-platform
set PLATFORM_REPO=https://github.com/OneOfficeautomation-Private-Limited/OneOfficeAutomation-Platform.git
rem ================================================================================================
title ONEOFFICEAUTOMATION + OOA Assistant

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. In PowerShell run:   winget install --id Git.Git -e
  echo then open a NEW window and run this file again.
  pause
  exit /b 1
)

rem ---- Java 21: use JAVA_HOME if it is set, else look in the usual install folders ----
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" goto have_java
set JAVA_HOME=
for /d %%d in ("%ProgramFiles%\Eclipse Adoptium\jdk-2*" "%ProgramFiles%\Microsoft\jdk-2*" "%ProgramFiles%\Java\jdk-2*" "%ProgramFiles%\Zulu\zulu-2*") do set "JAVA_HOME=%%~fd"
if not defined JAVA_HOME (
  echo Java 21 is not installed. In PowerShell run:   winget install --id EclipseAdoptium.Temurin.21.JDK -e
  echo then open a NEW window and run this file again.
  pause
  exit /b 1
)
:have_java
"%JAVA_HOME%\bin\java.exe" -version 2>&1 | findstr /c:"version \"2" >nul
if errorlevel 1 (
  echo The Java at "%JAVA_HOME%" is older than 21. Install Java 21:   winget install --id EclipseAdoptium.Temurin.21.JDK -e
  pause
  exit /b 1
)
echo Java: %JAVA_HOME%

rem ---- get or update the platform source ----
if not exist "%PLATFORM_SRC%\platform\gradlew.bat" (
  echo Downloading the platform to "%PLATFORM_SRC%" ...
  git clone --depth 1 "%PLATFORM_REPO%" "%PLATFORM_SRC%"
  if errorlevel 1 (
    echo Download failed. Check the internet connection and that you can open %PLATFORM_REPO%
    pause
    exit /b 1
  )
) else (
  echo Updating the platform in "%PLATFORM_SRC%" ...
  git -C "%PLATFORM_SRC%" pull --ff-only
)

rem ---- start both, each in its own window ----
echo Starting the OOA Assistant (port 8765) in a new window ...
start "OOA Assistant" cmd /k call "%~dp0start_assistant.bat"
echo Starting ONEOFFICEAUTOMATION (port 8720) in a new window; the first start builds it, wait for
echo "ONEOFFICEAUTOMATION is running" and note the one-time admin password it prints.
start "ONEOFFICEAUTOMATION" cmd /k call "%PLATFORM_SRC%\platform\scripts\start-ooa.bat"

echo.
echo ============================================================================================
echo  Command Center:  http://localhost:8720/          (sign in as admin with the one-time password)
echo  Agent Studio:    http://localhost:8720/studio/
echo  OOA Assistant:   http://127.0.0.1:8765/
echo.
echo  To let Studio's Assistant use YOUR trained model (once):
echo    Studio top bar: sun icon ^> One Model Connections... ^> + Connection
echo      Name: ooa     Provider: OpenAI-compatible     Model: ooa-assistant
echo      API key from: No key (local model)   Endpoint: http://127.0.0.1:8765/v1    tick Default, Save
echo    Then: Assistant ^> Build from a description ^> describe the job ^> Generate steps ^> Insert into the agent
echo  Close the two windows to stop everything.
echo ============================================================================================
pause
