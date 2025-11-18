#!/bin/bash
# Script to set up the python environment and install dependencies
# Utilizes uv Python package manager 

UV_INSTALLED=$(which uv || echo "")
if [ -z $UV_INSTALLED ]; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh    
fi

echo "Updating uv..."
uv self update

echo "Updating pip..."
pip install --upgrade pip

echo "Creating uv environment..."
uv venv --python 3.12 --seed
source .venv/bin/activate

echo "Manually install vllm..."
uv pip install vllm --torch-backend=auto

echo "Installing other dependencies..."
uv pip install -r requirements.txt