#!/bin/bash

# Python 3.11 이상 탐색 (높은 버전부터 순서대로)
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

# 위에서 못 찾았으면 python3 자체가 3.11 이상인지 확인
if [ -z "$PYTHON_CMD" ]; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        PYTHON_CMD=python3
    fi
fi

# 3.11 이상을 찾지 못하면 안내 후 종료
if [ -z "$PYTHON_CMD" ]; then
    CURRENT_VER=$(python3 --version 2>/dev/null || echo "설치되지 않음")
    echo ""
    echo "❌ Python 3.11 이상이 필요합니다. (현재: $CURRENT_VER)"
    echo ""
    echo "아래 명령어로 Python을 설치한 후 다시 start_mac.sh를 실행해주세요."
    echo ""
    echo "  1) Homebrew 설치 (없을 경우):"
    echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo ""
    echo "  2) Python 3.12 설치:"
    echo "     brew install python@3.12"
    echo ""
    exit 1
fi

echo "Python 버전 확인: $($PYTHON_CMD --version) ✅"

if [ ! -d "dodari_env" ]
then
    echo "처음 실행시 도다리 실행 환경부터 만들어야 합니다."
    echo "실행에 필요한 파일들을 설치합니다.."
    echo ""
    $PYTHON_CMD -m venv dodari_env
    . dodari_env/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo ""
        echo "환경 조성에 실패했습니다. 간혹 이럴때가 있습니다."
        echo "dodari_env 폴더를 제거하고 start_mac.sh를 다시 실행해보세요."
        deactivate
        exit 1
    fi

    echo ""
    echo "도다리 실행환경을 만드는데 성공했습니다!"
    echo ""
fi

. dodari_env/bin/activate

echo "Gemma4 API 서빙을 백그라운드로 시작합니다..."

# mlx_vlm이 설치된 Python의 절대경로를 환경변수로 저장한다
# dodari.py 내부에서 모델 교체 시 이 경로를 그대로 사용 → venv 격리 문제 해결
export MLX_PYTHON=$(python3 -c "import sys; print(sys.executable)")
echo "MLX Python 경로: $MLX_PYTHON"

python3 -m mlx_vlm.server --model mlx-community/gemma-4-e4b-it-8bit --kv-bits 8 --port 8000 &
SERVER_PID=$!

# 도다리가 종료될 때(Ctrl+C) API 서버도 함께 안전하게 종료되도록 설정
trap "echo 'API 서버(PID: $SERVER_PID)를 종료합니다...'; kill $SERVER_PID" EXIT

# 서버가 포트를 열고 준비할 수 있도록 잠시 대기
echo "API 서버 부팅 대기 중... (5초)"
sleep 5

echo "AI 번역기 도다리를 시작합니다."
echo "잠시만 기다려주세요.."
echo "다만 처음 실행시에는 AI모델을 설치해야해서 시간이 아주 오래걸립니다."

dodari_env/bin/python3 dodari.py
deactivate
