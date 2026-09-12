#!/bin/bash

# === Translation engine ===
# The engine is chosen inside the Dodari UI (model selector) and stored in
# ui_config.json under the "engine" key. This script only reads that value to
# decide whether the local model stack needs to be installed and started.
#   local      : local AI model (default)
#   claude-cli : Claude subscription via the claude CLI
#   codex-cli  : ChatGPT subscription via the Codex CLI
CONFIG_FILE="ui_config.json"
DODARI_ENGINE=""

read_engine_from_config() {
    [ -f "$CONFIG_FILE" ] || return 1
    python3 - "$CONFIG_FILE" <<'PYEOF' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1], encoding='utf-8') as fp:
        data = json.load(fp)
    engine = data.get('engine', '')
    if engine in ('local', 'claude-cli', 'codex-cli'):
        print(engine)
except Exception:
    pass
PYEOF
}

DODARI_ENGINE=$(read_engine_from_config)

[ -z "$DODARI_ENGINE" ] && DODARI_ENGINE="local"

echo "Translation engine: $DODARI_ENGINE"

# Search for Python 3.11+ (highest version first)
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11; do
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

    dodari_env/bin/pip install --upgrade pip --no-cache-dir
    dodari_env/bin/pip install -r requirements.txt --no-cache-dir
    # MLX is only needed for the local model engine — CLI engines skip the heavy install
    if [ "$DODARI_ENGINE" = "local" ]; then
        dodari_env/bin/pip install mlx-vlm==0.5.0 mlx==0.31.2 --no-cache-dir 2>/dev/null || true
    fi

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

# CLI subscription engines translate through the CLI subprocess,
# so no local model download and no local API server are needed.
if [ "$DODARI_ENGINE" = "local" ]; then
    echo "Starting Gemma4 API server in the background..."

    # Save the absolute path of the Python with mlx_vlm installed
    # Used by dodari.py when switching models → avoids venv isolation issues
    export MLX_PYTHON=$(dodari_env/bin/python3 -c "import sys; print(sys.executable)")
    echo "MLX Python path: $MLX_PYTHON"

    MLX_MODEL="mlx-community/gemma-4-31b-it-4bit"
    # Reuse the already-downloaded model when the HuggingFace cache holds a complete copy — never re-download.
    # Only the server process gets HF_HUB_OFFLINE; dodari.py itself stays online (Docling downloads its own models).
    MLX_ENV=""
    if dodari_env/bin/python3 - "$MLX_MODEL" <<'PYEOF'
import json, os, sys
model = sys.argv[1]
hub = os.environ.get('HF_HUB_CACHE') or os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub')
repo = os.path.join(hub, 'models--' + model.replace('/', '--'))
try:
    sha = open(os.path.join(repo, 'refs', 'main'), encoding='utf-8').read().strip()
    snap = os.path.join(repo, 'snapshots', sha)
    idx = os.path.join(snap, 'model.safetensors.index.json')
    need = set(json.load(open(idx, encoding='utf-8'))['weight_map'].values()) if os.path.exists(idx) else {'model.safetensors'}
    need.add('config.json')
    ok = all(os.path.exists(os.path.realpath(os.path.join(snap, f))) for f in need)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYEOF
    then
        MLX_ENV="HF_HUB_OFFLINE=1"
        echo "Model $MLX_MODEL found in the local HuggingFace cache — offline mode, no re-download."
    else
        echo "Model $MLX_MODEL is not in the local cache yet — it will be downloaded from HuggingFace (progress below)."
    fi

    env $MLX_ENV dodari_env/bin/python3 -m mlx_vlm.server --model "$MLX_MODEL" --kv-bits 8 --port 8000 &
    SERVER_PID=$!

    # Shut down the API server safely when Dodari exits (Ctrl+C)
    trap "echo 'Stopping API server (PID: $SERVER_PID)...'; kill $SERVER_PID" EXIT

    # Wait for the server to open its port
    echo "Waiting for API server to boot... (5 seconds)"
    sleep 5
else
    echo "Skipping local model server (using $DODARI_ENGINE)."
fi

echo "Starting Dodari AI Translator."
echo "Please wait..."

dodari_env/bin/python3 dodari.py
deactivate
