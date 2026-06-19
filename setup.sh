#!/usr/bin/env bash
set -e

python3 -m venv venv

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

pip install -r requirements.txt
