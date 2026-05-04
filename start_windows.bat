@echo off

rem === 1단계: Python 3.11+ 확인 ===
echo [1/6] Python 버전 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [오류] Python이 설치되어 있지 않습니다.
    echo Python 3.11 이상을 설치한 후 다시 실행해주세요.
    echo 다운로드: https://www.python.org/downloads/
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
    echo [오류] Python 3.11 이상이 필요합니다. 현재 버전: %PY_VER%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 11 (
    echo [오류] Python 3.11 이상이 필요합니다. 현재 버전: %PY_VER%
    pause
    exit /b 1
)
echo   완료 - Python %PY_VER%

rem === 2단계: Ollama 설치 확인 ===
echo.
echo [2/6] Ollama 설치 확인 중...
where ollama >nul 2>&1
if not errorlevel 1 goto OLLAMA_OK

echo   Ollama가 설치되어 있지 않습니다. 자동 설치를 시작합니다.
echo   설치 파일 크기는 약 1.8GB입니다. 인터넷 속도에 따라 수분 이상 소요될 수 있습니다.
echo   아래에 진행 상황이 표시됩니다. 창이 멈춘 것처럼 보여도 정상이니 기다려주세요.
echo.
winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
if not errorlevel 1 goto OLLAMA_PATH_CHECK

echo.
echo   winget 실패. curl로 직접 다운로드합니다... (약 1.8GB)
curl -L --progress-bar -o "%TEMP%\OllamaSetup.exe" "https://ollama.com/download/OllamaSetup.exe"
if errorlevel 1 (
    echo.
    echo [오류] 다운로드에 실패했습니다.
    echo 수동 설치 후 다시 실행해주세요: https://ollama.com/download
    pause
    exit /b 1
)
echo   설치 프로그램을 실행합니다...
"%TEMP%\OllamaSetup.exe" /silent
timeout /t 15 /nobreak >nul

:OLLAMA_PATH_CHECK
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo [안내] Ollama 설치 완료. PATH 적용을 위해 재시작이 필요합니다.
    echo 이 창을 닫고 새 명령 프롬프트에서 start_windows.bat을 다시 실행해주세요.
    pause
    exit /b 1
)

:OLLAMA_OK
echo   완료 - Ollama 확인됨

rem === 3단계: gemma4:e4b 기본 모델 확인 ===
echo.
echo [3/6] 기본 AI 모델(gemma4:e4b) 확인 중...
ollama list 2>nul | findstr /i "gemma4:e4b" >nul
if not errorlevel 1 goto MODEL_OK

echo   기본 AI 모델(gemma4:e4b)을 다운로드합니다. (약 3GB, 시간이 걸립니다)
echo   고품질 모델(gemma4:31b)은 UI에서 선택 시 Ollama가 자동으로 처리합니다.
echo   다운로드 진행률이 표시됩니다. 완료까지 기다려주세요.
echo.
ollama pull gemma4:e4b
if errorlevel 1 (
    echo.
    echo [오류] gemma4:e4b 모델 다운로드에 실패했습니다.
    echo 인터넷 연결을 확인하고 다시 시도해주세요.
    pause
    exit /b 1
)

:MODEL_OK
echo   완료 - gemma4:e4b 준비됨

rem === 4단계: Python 가상환경 확인 ===
echo.
echo [4/6] Python 가상환경 확인 중...
if exist "%~dp0\dodari_env\Scripts\activate.bat" goto VENV_OK

echo   가상환경이 없습니다. 새로 생성합니다...
python -m venv dodari_env
if errorlevel 1 (
    echo.
    echo [오류] 가상환경 생성에 실패했습니다.
    pause
    exit /b 1
)

echo   필요한 패키지를 설치합니다. 첫 실행 시 수분이 소요됩니다...
cd /d "%~dp0\dodari_env\Scripts"
call activate.bat
cd /d "%~dp0"
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [오류] 패키지 설치에 실패했습니다.
    echo dodari_env 폴더를 삭제한 후 start_windows.bat을 다시 실행해주세요.
    call dodari_env\Scripts\deactivate.bat 2>nul
    pause
    exit /b 1
)
call dodari_env\Scripts\deactivate.bat 2>nul

:VENV_OK
echo   완료 - 가상환경 준비됨

rem === 5단계: Ollama 서버 실행 ===
echo.
echo [5/6] Ollama 서버 시작 중...
tasklist /fi "imagename eq ollama.exe" 2>nul | findstr /i "ollama.exe" >nul
if not errorlevel 1 (
    echo   완료 - Ollama 서버가 이미 실행 중입니다.
    goto OLLAMA_SERVER_OK
)
start /B ollama serve
echo   서버 초기화 대기 중 (5초)...
timeout /t 5 /nobreak >nul

:OLLAMA_SERVER_OK
echo   완료 - Ollama 서버 준비됨

rem === 6단계: 도다리 앱 실행 ===
echo.
echo [6/6] pdf 인식툴을 설치중입니다. 조금만 더 기다려 주세요.
echo.
set PYTHON="%~dp0\dodari_env\Scripts\Python.exe"
%PYTHON% dodari.py

if errorlevel 1 (
    echo.
    echo [오류] 도다리 실행 중 오류가 발생했습니다.
    pause
)
