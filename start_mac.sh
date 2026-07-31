#!/bin/bash

# === Translation engine selection ===
# Asked once on first run, then stored in ui_config.json ("engine" key).
#   local      : local AI model (existing behaviour, downloads the model)
#   claude-cli : your own Claude subscription via the official claude CLI
#   codex-cli  : your own ChatGPT subscription via the official Codex CLI
# CLI engines run on YOUR account and YOUR subscription limits.
# You install and log in to the CLI yourself; Dodari never provides an account.
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

save_engine_to_config() {
    python3 - "$CONFIG_FILE" "$1" <<'PYEOF' 2>/dev/null
import json, os, sys
path, engine = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    try:
        with open(path, encoding='utf-8') as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}
data['engine'] = engine
with open(path, 'w', encoding='utf-8') as fp:
    json.dump(data, fp, ensure_ascii=False)
PYEOF
}

DODARI_ENGINE=$(read_engine_from_config)

if [ -z "$DODARI_ENGINE" ]; then
    echo ""
    echo "=================================================="
    echo " Choose a translation engine (asked only once)"
    echo "=================================================="
    echo "  1) Local AI model      - free, downloads a model, needs a capable Mac"
    echo "  2) Claude subscription - uses YOUR Claude account (Pro/Max)"
    echo "  3) ChatGPT subscription- uses YOUR ChatGPT account (Plus/Pro)"
    echo ""
    echo "  Options 2 and 3 run on your own account and your own"
    echo "  subscription limits. The CLI is installed and logged in by you."
    echo ""
    printf "Enter 1, 2 or 3 [1]: "
    read -r ENGINE_CHOICE
    case "$ENGINE_CHOICE" in
        2) DODARI_ENGINE="claude-cli" ;;
        3) DODARI_ENGINE="codex-cli" ;;
        *) DODARI_ENGINE="local" ;;
    esac
    save_engine_to_config "$DODARI_ENGINE"
    echo "Selected engine: $DODARI_ENGINE (saved to $CONFIG_FILE)"
    echo ""
fi

echo "Translation engine: $DODARI_ENGINE"

# Install and log in to the selected CLI (only for the subscription engines)
setup_cli_engine() {
    local bin_name="$1" install_cmd="$2" login_cmd="$3"

    if ! command -v "$bin_name" &>/dev/null; then
        echo ""
        echo "$bin_name CLI is not installed. Installing now..."
        eval "$install_cmd"
        hash -r
        # Fresh installs often land in these locations before PATH is refreshed
        for extra in "$HOME/.local/bin" "$HOME/.claude/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
            [ -d "$extra" ] && PATH="$extra:$PATH"
        done
        export PATH
        if ! command -v "$bin_name" &>/dev/null; then
            echo ""
            echo "Could not find $bin_name after installation."
            echo "Open a new terminal and run start_mac.sh again,"
            echo "or install it manually with: $install_cmd"
            exit 1
        fi
    fi
    echo "$bin_name CLI found: $(command -v "$bin_name")"

    if ! $login_cmd; then
        echo ""
        echo "Login check failed for $bin_name. Please complete the login and rerun."
        exit 1
    fi
}

# Returns 0 when the claude CLI is logged in with a subscription account
claude_login_ok() {
    local out
    out=$(printf 'ok' | claude -p --output-format json --tools "" \
        --disable-slash-commands --strict-mcp-config --settings '{}' 2>/dev/null)
    case "$out" in
        *'"is_error":false'*) return 0 ;;
    esac

    echo ""
    echo "You are not logged in to the claude CLI yet."
    echo "A browser window will open for a one-time login."
    echo "Finish the login, then return to this window."
    echo ""
    claude /login || claude
    out=$(printf 'ok' | claude -p --output-format json --tools "" \
        --disable-slash-commands --strict-mcp-config --settings '{}' 2>/dev/null)
    case "$out" in
        *'"is_error":false'*) return 0 ;;
    esac
    return 1
}

codex_login_ok() {
    if codex login status >/dev/null 2>&1; then
        return 0
    fi
    echo ""
    echo "You are not logged in to the Codex CLI yet."
    echo "A browser window will open for a one-time login."
    echo "Finish the login, then return to this window."
    echo ""
    codex login || return 1
    codex login status >/dev/null 2>&1
}

if [ "$DODARI_ENGINE" = "claude-cli" ]; then
    setup_cli_engine "claude" "curl -fsSL https://claude.ai/install.sh | bash" claude_login_ok
elif [ "$DODARI_ENGINE" = "codex-cli" ]; then
    setup_cli_engine "codex" "npm install -g @openai/codex" codex_login_ok
fi

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

    dodari_env/bin/python3 -m mlx_vlm.server --model mlx-community/gemma-4-31b-it-4bit --kv-bits 8 --port 8000 &
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
