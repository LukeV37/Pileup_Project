#!/bin/bash
WORK_DIR=$(pwd)
cd software
python3 -m venv torch
if test -f ./torch/bin/activate; then
    cd torch
    source ./bin/activate
    pip install --upgrade pip
    pip install -r pip_requirements.txt
else
    echo "Virtual Environment could not be created..."
fi
cd $WORK_DIR
