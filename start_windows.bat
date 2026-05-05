@echo off
chcp 65001 >nul

rem === Step 1: Check Python 3.11+ ===
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [Error] Python is not installed.
    echo Please install Python 3.11 or higher and run this script again.
    echo Download: https://www.python.org/downloads/
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo [Error] Python 3.11 or higher is required. Current: %PY_VER%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 11 (
    echo [Error] Python 3.11 or higher is required. Current: %PY_VER%
    pause
    exit /b 1
)
echo   Done - Python %PY_VER%

rem === Step 2: Check Ollama installation ===
echo.
echo [2/6] Checking Ollama installation...
where ollama >nul 2>&1
if not errorlevel 1 goto OLLAMA_OK

echo   Ollama is not installed. Starting automatic installation.
echo   Download size is approximately 1.8GB. This may take a while depending on your connection.
echo   Progress will be shown below. Please wait until the window closes on its own.
echo.
winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
if not errorlevel 1 goto OLLAMA_PATH_CHECK

echo.
echo   winget failed. Downloading via curl... (approx. 1.8GB)
curl -L --progress-bar -o "%TEMP%\OllamaSetup.exe" "https://ollama.com/download/OllamaSetup.exe"
if errorlevel 1 (
    echo.
    echo [Error] Download failed.
    echo Please install manually and run this script again: https://ollama.com/download
    pause
    exit /b 1
)
echo   Running installer...
"%TEMP%\OllamaSetup.exe" /silent
timeout /t 15 /nobreak >nul

:OLLAMA_PATH_CHECK
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo [Notice] Ollama installation complete. PATH refresh requires a new terminal session.
    echo Please close this window and run start_windows.bat again from a new terminal.
    pause
    exit /b 1
)

:OLLAMA_OK
echo   Done - Ollama confirmed

rem === Step 3: Check base AI model (gemma4:e4b) ===
echo.
echo [3/6] Checking base AI model (gemma4:e4b)...
ollama list 2>nul | findstr /i "gemma4:e4b" >nul
if not errorlevel 1 goto MODEL_OK

echo   Downloading base AI model (gemma4:e4b). (approx. 3GB, this will take a while)
echo   The high-quality model (gemma4:31b) can be selected in the UI — Ollama handles it automatically.
echo   Download progress is shown below. Please wait until complete.
echo.
ollama pull gemma4:e4b
if errorlevel 1 (
    echo.
    echo [Error] gemma4:e4b model download failed.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

:MODEL_OK
echo   Done - gemma4:e4b ready

rem === Step 4: Check Python virtual environment ===
echo.
echo [4/6] Checking Python virtual environment...
if exist "%~dp0\dodari_env\Scripts\activate.bat" goto VENV_OK

echo   Creating virtual environment...
python -m venv dodari_env
if errorlevel 1 (
    echo.
    echo [Error] Virtual environment creation failed.
    pause
    exit /b 1
)

echo   Installing required packages. This may take a few minutes on first run...
cd /d "%~dp0\dodari_env\Scripts"
call activate.bat
cd /d "%~dp0"
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [Error] Package installation failed.
    echo Delete the dodari_env folder and run start_windows.bat again.
    call dodari_env\Scripts\deactivate.bat 2>nul
    pause
    exit /b 1
)
call dodari_env\Scripts\deactivate.bat 2>nul

:VENV_OK
echo   Done - virtual environment ready

rem === Step 5: Start Ollama server ===
echo.
echo [5/6] Starting Ollama server...
tasklist /fi "imagename eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul
if not errorlevel 1 (
    echo   Done - Ollama server is already running.
    goto OLLAMA_SERVER_OK
)
start /B ollama serve
echo   Waiting for server initialization... (5 seconds)
timeout /t 5 /nobreak >nul

:OLLAMA_SERVER_OK
echo   Done - Ollama server ready

rem === Step 6: Start Dodari ===
echo.
echo [6/6] Installing PDF recognition tool. Please wait a moment...
echo.
set PYTHON="%~dp0\dodari_env\Scripts\Python.exe"
%PYTHON% dodari.py

if errorlevel 1 (
    echo.
    echo [Error] An error occurred while running Dodari.
    pause
)
