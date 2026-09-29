#!/bin/bash
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"

# ============================================================
# start_ubuntu.sh — Dodari (Ubuntu + RTX 4090 / vLLM)
# ============================================================
# Environment: Ubuntu 22.04+, CUDA 12.1+, RTX 4090 (VRAM 24GB)
# Model: cyankiwi/gemma-4-31B-it-AWQ-4bit (AWQ 4-bit, bfloat16)
# ============================================================

# Run from this script's folder so relative paths work from a desktop launcher
cd "$(dirname "$0")" || exit 1

# >>> dodari_config
# Model settings: the defaults below, overridden by the "vllm" section of dodari_config.json next to this
# script when that file exists (your own settings file, not tracked by git; dodari.py reads the same file).
# Missing file, broken JSON or missing keys keep the defaults below.
HF_MODEL_ID="cyankiwi/gemma-4-31B-it-AWQ-4bit"
MODEL_PATH="./models"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="3072"
if [ -f ./dodari_config.json ] && command -v python3 >/dev/null 2>&1; then
    _cfg=$(python3 -c '
import json, sys
try:
    v = json.load(open(sys.argv[1], encoding="utf-8")).get("vllm") or {}
except Exception:
    v = {}
for k in ("model_id", "model_path", "gpu_memory_utilization", "max_model_len"):
    print(v.get(k) if v.get(k) not in (None, "") else "")
' ./dodari_config.json 2>/dev/null)
    _c_id=$(printf '%s\n' "$_cfg" | sed -n 1p)
    _c_path=$(printf '%s\n' "$_cfg" | sed -n 2p)
    _c_util=$(printf '%s\n' "$_cfg" | sed -n 3p)
    _c_len=$(printf '%s\n' "$_cfg" | sed -n 4p)
    [ -n "$_c_id" ] && HF_MODEL_ID="$_c_id"
    [ -n "$_c_path" ] && MODEL_PATH="$_c_path"
    [ -n "$_c_util" ] && GPU_MEM_UTIL="$_c_util"
    [ -n "$_c_len" ] && MAX_MODEL_LEN="$_c_len"
fi
# <<< dodari_config

# Translation engine saved by the Dodari UI in ui_config.local.json ("engine" key, not tracked by git), read the same way as start_mac.sh
# Until Dodari moves an edited ui_config.json into ui_config.local.json on its first start, ui_config.json is read instead
# Subscription CLI engines (claude-cli, codex-cli) skip the vLLM install, the model download (about 20GB) and the vLLM server
# DODARI_SETUP_ONLY=1 installs the environment only and does not start Dodari (README.md "For AI assistants: how to install Dodari", step 4)
CONFIG_FILE="ui_config.local.json"
LEGACY_CONFIG_FILE="ui_config.json"
DODARI_ENGINE=""
if command -v python3 >/dev/null 2>&1; then
    DODARI_ENGINE=$(python3 - "$CONFIG_FILE" "$LEGACY_CONFIG_FILE" <<'PYEOF' 2>/dev/null
import json, os, sys
try:
    path = sys.argv[1] if os.path.exists(sys.argv[1]) else sys.argv[2]
    with open(path, encoding='utf-8') as fp:
        engine = json.load(fp).get('engine', '')
    if engine in ('local', 'claude-cli', 'codex-cli'):
        print(engine)
except Exception:
    pass
PYEOF
)
fi
[ -z "$DODARI_ENGINE" ] && DODARI_ENGINE="local"

echo "=============================================="
if [ "$DODARI_ENGINE" = "local" ]; then
    echo " Dodari (Ubuntu / vLLM mode)"
    echo " Model: $HF_MODEL_ID"
else
    echo " Dodari (Ubuntu / $DODARI_ENGINE)"
fi
echo "=============================================="
echo ""

# ----------------------------------------------------------
# 1. Virtual environment setup (first run only)
# ----------------------------------------------------------
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11; do
    if command -v $cmd >/dev/null 2>&1; then
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

    if [ "$DODARI_ENGINE" = "local" ]; then
        echo ""
        echo "Base packages installed. Installing vLLM..."
        echo "(May take 10–20 minutes depending on your CUDA version)"
        echo ""

        # Install vLLM (CUDA 12.1 baseline, includes PyTorch)
        # For other CUDA versions: https://docs.vllm.ai/en/latest/getting_started/installation.html
        dodari_env/bin/pip install vllm==0.21.0 "huggingface_hub[cli]>=1.11.0,<2.0" --no-cache-dir

        if [ $? -ne 0 ]; then
            echo ""
            echo "[Error] vLLM / huggingface_hub installation failed."
            echo "Check CUDA version: nvidia-smi | nvcc --version"
            echo "Manual install: pip install vllm 'huggingface_hub[cli]>=1.11.0'"
            deactivate
            exit 1
        fi
    else
        echo "Skipping vLLM install (using $DODARI_ENGINE)."
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
if [ "$DODARI_ENGINE" != "local" ]; then
    echo "Skipping model download (using $DODARI_ENGINE)."
    echo ""
elif [ ! -d "$MODEL_PATH" ]; then
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

if [ "$DODARI_SETUP_ONLY" = "1" ]; then
    echo "Setup complete (DODARI_SETUP_ONLY=1): Dodari was not started."
    deactivate
    exit 0
fi

# Subscription CLI engines start Dodari directly, without a local model server
if [ "$DODARI_ENGINE" != "local" ]; then
    echo "Skipping vLLM server (using $DODARI_ENGINE)."
    echo "Starting Dodari."
    echo ""
    dodari_env/bin/python3 dodari.py
    deactivate
    exit 0
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

# Multi-GPU: set VLLM_TP to the number of GPUs (tensor parallelism, issue #17)
# Unquantized (e.g. BF16) models: set VLLM_QUANT=none so vLLM auto-detects from the model config
export VLLM_TP="${VLLM_TP:-1}"
export VLLM_QUANT="${VLLM_QUANT:-compressed-tensors}"

QUANT_OPT=""
if [ -n "$VLLM_QUANT" ] && [ "$VLLM_QUANT" != "none" ] && [ "$VLLM_QUANT" != "auto" ]; then
    QUANT_OPT="--quantization $VLLM_QUANT"
fi

# Start vLLM server (logs printed directly to terminal)
dodari_env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$HF_MODEL_ID" \
    $QUANT_OPT \
    --dtype bfloat16 \
    --tensor-parallel-size "$VLLM_TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
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
