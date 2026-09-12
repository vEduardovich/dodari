@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem === Step 0: Read translation engine saved by the Dodari UI (ui_config.json) ===
rem The engine is chosen inside the Dodari UI (model selector), never here.
rem   local      : local AI model (default, installs Ollama + model)
rem   claude-cli : Claude subscription via the claude CLI
rem   codex-cli  : ChatGPT subscription via the Codex CLI
set CONFIG_FILE=%~dp0ui_config.json
set DODARI_ENGINE=

rem Python is needed to read the config, so locate it first (full check happens in Step 1)
set CFG_PY=
python --version >nul 2>&1
if not errorlevel 1 set CFG_PY=python
if "%CFG_PY%"=="" (
    py --version >nul 2>&1
    if not errorlevel 1 set CFG_PY=py
)
if "%CFG_PY%"=="" (
    echo.
    echo [Error] Python is not installed.
    echo Please install Python 3.11 or higher and run this script again.
    echo Download: https://www.python.org/downloads/
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

if exist "%CONFIG_FILE%" (
    for /f "usebackq delims=" %%e in (`%CFG_PY% -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));e=d.get('engine','');print(e if e in ('local','claude-cli','codex-cli') else '')" "%CONFIG_FILE%" 2^>nul`) do set DODARI_ENGINE=%%e
)

if "%DODARI_ENGINE%"=="" set DODARI_ENGINE=local

echo Translation engine: %DODARI_ENGINE%

rem === Step 1: Check Python 3.11+ ===
echo [1/6] Checking Python installation...
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" (
    py --version >nul 2>&1
    if not errorlevel 1 set PYTHON_CMD=py
)
if "%PYTHON_CMD%"=="" (
    echo.
    echo [Error] Python is not installed.
    echo Please install Python 3.11 or higher and run this script again.
    echo Download: https://www.python.org/downloads/
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VER=%%v
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

rem CLI subscription engines translate through the CLI, so Ollama and the
rem local model download (Steps 2, 3 and 5) are skipped entirely.
if not "%DODARI_ENGINE%"=="local" (
    echo.
    echo [2/6] Skipping Ollama install ^(using %DODARI_ENGINE%^)
    echo [3/6] Skipping local model download ^(using %DODARI_ENGINE%^)
    goto VENV_STEP
)

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
echo   The high-quality model (gemma4:31b) can be selected in the UI -- Ollama handles it automatically.
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

:VENV_STEP
rem === Step 4: Check Python virtual environment ===
echo.
echo [4/6] Checking Python virtual environment...
if exist "%~dp0\dodari_env\Scripts\activate.bat" goto VENV_OK

echo   Creating virtual environment...
%PYTHON_CMD% -m venv dodari_env
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
dodari_env\Scripts\pip.exe install --upgrade pip --no-cache-dir
dodari_env\Scripts\pip.exe install -r requirements.txt --no-cache-dir
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
if not "%DODARI_ENGINE%"=="local" (
    echo.
    echo [5/6] Skipping Ollama server ^(using %DODARI_ENGINE%^)
    goto OLLAMA_SERVER_OK
)

tasklist /fi "imagename eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul
if not errorlevel 1 (
    echo   Done - Ollama server is already running.
    goto OLLAMA_SERVER_OK
)
start /B ollama serve
echo   Waiting for server initialization... (5 seconds)
timeout /t 5 /nobreak >nul

:OLLAMA_SERVER_OK
if "%DODARI_ENGINE%"=="local" echo   Done - Ollama server ready

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
