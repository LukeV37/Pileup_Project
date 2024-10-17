#!/bin/bash
mkdir -p plots/preprocessing
nohup python3 -u preprocessing.py > preprocessing.log 2>&1 &
