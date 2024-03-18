@echo off

if not exist "%~dp0\venv\Scripts" (
    echo 처음 실행시 도다리 실행 환경부터 만들어야 합니다.
    echo 실행에 필요한 파일들을 설치합니다.. 시간이 꽤 오래걸립니다.
    python -m venv venv
	cd /d "%~dp0\venv\Scripts"
    call activate.bat

    cd /d "%~dp0"
    pip install -r requirements.txt

    echo.
    echo 도다리 실행환경을 만드는데 성공했습니다!
    echo.
)

if errorlevel 1 (
    echo.
    echo 환경 조성에 실패했습니다. 간혹 이럴때가 있습니다.
    echo "venv 폴더를 제거하고 start_windows.bat를 다시 실행해보세요.
    pause
)

echo 대용량 번역기 도다리를 시작합니다.
echo 잠시만 기다려주세요..
goto :activate_venv

:launch
%PYTHON% dodari.py
pause

:activate_venv
set PYTHON="%~dp0\venv\Scripts\Python.exe"
goto :launch
:endofscript

echo.
echo 실행에 실패했습니다. 창을 닫습니다.
pause