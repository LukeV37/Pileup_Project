#!/bin/bash
mkdir -p plots/Mfrac_regression
nohup python3 -u Jet_Attention_Model.py > train_Mfrac.log 2>&1 &
