#!/bin/bash

# Search for Python 3.11+ (highest version first)
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

# If not found above, check if python3 itself is 3.11+
if [ -z "$PYTHON_CMD" ]; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        PYTHON_CMD=python3
    fi
fi

# If no 3.11+ found, show instructions and exit
if [ -z "$PYTHON_CMD" ]; then
    CURRENT_VER=$(python3 --version 2>/dev/null || echo "not installed")
    echo ""
    echo "❌ Python 3.11 or higher is required. (current: $CURRENT_VER)"
    echo ""
    echo "Install Python using the commands below, then run this script again:"
    echo ""
    echo "  1) Install Homebrew (if not installed):"
    echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo ""
    echo "  2) Install Python 3.12:"
    echo "     brew install python@3.12"
    echo ""
    exit 1
fi

echo "Python version check: $($PYTHON_CMD --version) ✅"

if [ ! -d "dodari_env" ]
then
    echo "First run: setting up Dodari environment."
    echo "Installing required packages..."
    echo ""
    $PYTHON_CMD -m venv dodari_env
    . dodari_env/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mlx-vlm mlx 2>/dev/null || true

    if [ $? -ne 0 ]; then
        echo ""
        echo "Environment setup failed."
        echo "Delete the dodari_env folder and run start_mac.sh again."
        deactivate
        exit 1
    fi

    echo ""
    echo "Dodari environment created successfully!"
    echo ""
fi

. dodari_env/bin/activate

echo "Starting Gemma4 API server in the background..."

# Save the absolute path of the Python with mlx_vlm installed
# Used by dodari.py when switching models → avoids venv isolation issues
export MLX_PYTHON=$(python3 -c "import sys; print(sys.executable)")
echo "MLX Python path: $MLX_PYTHON"

python3 -m mlx_vlm.server --model mlx-community/gemma-4-31b-it-4bit --kv-bits 8 --port 8000 &
SERVER_PID=$!

# Shut down the API server safely when Dodari exits (Ctrl+C)
trap "echo 'Stopping API server (PID: $SERVER_PID)...'; kill $SERVER_PID" EXIT

# Wait for the server to open its port
echo "Waiting for API server to boot... (5 seconds)"
sleep 5

echo "Starting Dodari AI Translator."
echo "Please wait..."

dodari_env/bin/python3 dodari.py
deactivate
