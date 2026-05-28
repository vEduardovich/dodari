#!/bin/bash

# ============================================================
# start_ubuntu.sh — Dodari (Ubuntu + RTX 4090 / vLLM)
# ============================================================
# Environment: Ubuntu 22.04+, CUDA 12.1+, RTX 4090 (VRAM 24GB)
# Model: cyankiwi/gemma-4-31B-it-AWQ-4bit (AWQ 4-bit, bfloat16)
# ============================================================

HF_MODEL_ID="cyankiwi/gemma-4-31B-it-AWQ-4bit"
MODEL_PATH="./models"

echo "=============================================="
echo " Dodari (Ubuntu / vLLM mode)"
echo " Model: $HF_MODEL_ID"
echo "=============================================="
echo ""

# ----------------------------------------------------------
# 1. Virtual environment setup (first run only)
# ----------------------------------------------------------
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        PYTHON_CMD=python3
    fi
fi
if [ -z "$PYTHON_CMD" ]; then
    CURRENT_VER=$(python3 --version 2>/dev/null || echo "not installed")
    echo ""
    echo "[Error] Python 3.11 or higher is required. (current: $CURRENT_VER)"
    echo ""
    echo "Install: sudo apt install python3.11 python3.11-venv"
    echo ""
    exit 1
fi
echo "Python version: $($PYTHON_CMD --version)"
echo ""

if [ ! -d "dodari_env" ]; then
    echo "First run: setting up Dodari environment."
    echo "Installing required packages... (this may take a few minutes)"
    echo ""

    $PYTHON_CMD -m venv dodari_env
    . dodari_env/bin/activate

    # Install base packages (requirements.txt — no mlx, Ubuntu compatible)
    dodari_env/bin/pip install -r requirements.txt --no-cache-dir

    if [ $? -ne 0 ]; then
        echo ""
        echo "[Error] Base package installation failed."
        echo "Delete the dodari_env folder and run start_ubuntu.sh again."
        deactivate
        exit 1
    fi

    echo ""
    echo "Base packages installed. Installing vLLM..."
    echo "(May take 10–20 minutes depending on your CUDA version)"
    echo ""

    # Install vLLM (CUDA 12.1 baseline, includes PyTorch)
    # For other CUDA versions: https://docs.vllm.ai/en/latest/getting_started/installation.html
    dodari_env/bin/pip install vllm "huggingface_hub[cli]>=1.11.0" --no-cache-dir

    if [ $? -ne 0 ]; then
        echo ""
        echo "[Error] vLLM / huggingface_hub installation failed."
        echo "Check CUDA version: nvidia-smi | nvcc --version"
        echo "Manual install: pip install vllm 'huggingface_hub[cli]>=1.11.0'"
        deactivate
        exit 1
    fi

    echo ""
    echo "Dodari environment created successfully!"
    echo ""
else
    . dodari_env/bin/activate
fi

# ----------------------------------------------------------
# 2. Model download (first run only)
# ----------------------------------------------------------
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading model: $HF_MODEL_ID"
    echo "Save path: $MODEL_PATH"
    echo "(May take several minutes to hours — approx. 20GB)"
    echo ""

    hf download "$HF_MODEL_ID" --local-dir "$MODEL_PATH"

    if [ $? -ne 0 ]; then
        echo ""
        echo "[Error] Model download failed."
        echo "Check HuggingFace login: hf login"
        deactivate
        exit 1
    fi

    echo ""
    echo "Model download complete!"
    echo ""
else
    echo "Model already exists: $MODEL_PATH (skipping download)"
    echo ""
fi

# ----------------------------------------------------------
# 3. Start vLLM API server (background)
# ----------------------------------------------------------
echo "Starting vLLM API server in the background..."
echo "Model path: $MODEL_PATH"
echo ""

# Save the absolute path of the Python with vLLM installed
# Used by dodari.py when switching models → avoids venv isolation issues
export VLLM_PYTHON=$(dodari_env/bin/python3 -c "import sys; print(sys.executable)")
# Export local model path so dodari.py's reload_llm_server uses the same path
export VLLM_MODEL="$MODEL_PATH"
echo "vLLM Python path: $VLLM_PYTHON"

# Prevent memory fragmentation — reuse PyTorch reserved memory without fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Start vLLM server (logs printed directly to terminal)
dodari_env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$HF_MODEL_ID" \
    --quantization compressed-tensors \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --port 8000 &

SERVER_PID=$!

# Shut down the vLLM server safely when Dodari exits (Ctrl+C)
trap "echo ''; echo 'Stopping vLLM server (PID: $SERVER_PID)...'; kill $SERVER_PID 2>/dev/null; deactivate" EXIT

# ----------------------------------------------------------
# 4. Start Dodari
# ----------------------------------------------------------
echo ""
echo "Starting Dodari while the vLLM server boots up."
echo "Translation will be available once the server is ready."
echo ""

dodari_env/bin/python3 dodari.py

deactivate
