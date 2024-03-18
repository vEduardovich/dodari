#!/bin/bash

if [ ! -d "venv" ]
then
    echo "처음 실행시 도다리 실행 환경부터 만들어야 합니다."
    echo "실행에 필요한 파일들을 설치합니다.."
    echo ""
    python -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo ""
        echo "환경 조성에 실패했습니다. 간혹 이럴때가 있습니다."
        echo "venv 폴더를 제거하고 install.sh를 다시 실행해보세요."
        deactivate
        exit 1
    fi

    echo ""
    echo "도다리 실행환경을 만드는데 성공했습니다!"
    echo ""
fi

source venv/bin/activate
echo "대용량 번역기 도다리를 시작합니다."
echo "잠시만 기다려주세요.."

PYTHON="venv/bin/python"
python dodari.py

deactivate