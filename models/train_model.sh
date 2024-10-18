#!/bin/bash
mkdir -p plots/regression
nohup python3 -u Jet_Attention_Model.py > train.log 2>&1 &
