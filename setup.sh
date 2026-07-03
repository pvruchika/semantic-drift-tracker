#!/usr/bin/env bash
set -e

if command -v python3 >/dev/null 2>&1 && python3 -c "" >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Placed outside the repo to avoid Windows' 260-char path limit, which
# jupyterlab's bundled static assets exceed when nested under a long repo path.
VENV_DIR="${VENV_DIR:-/c/venvs/sdt}"

"$PYTHON" -m venv "$VENV_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    source "$VENV_DIR/Scripts/activate"
fi

pip install -r requirements.txt

echo "Virtualenv ready at $VENV_DIR. Activate it with: source $VENV_DIR/Scripts/activate"
