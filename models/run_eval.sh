#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 4 agruments"
    echo "1: Path to Efrac Model (results/<Efrac_model>.torch"
    echo "2: Path to Mfrac Model (results/<Mfrac_model>.torch"
    echo "3: Path to Input File (../pythia/output/<file>.root)"
    echo "4: Path to Output File (data/<file>.root)"
    exit 1
fi

Efrac_model=$1
Mfrac_model=$2
in_file=$3
out_file=$4

nohup python -u eval_scores.py $Efrac_model $Mfrac_model $in_file $out_file > eval_scores.log 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f eval_scores.log
