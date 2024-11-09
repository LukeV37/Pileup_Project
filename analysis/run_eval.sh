#!/bin/bash

if [ -z "$7" ]; then
    echo "Must enter 7 agruments"
    echo "1: Path to Input File (../pythia/output/<file>.root)"
    echo "2: Path to Efrac Model (../models/results/<Efrac>.torch)"
    echo "3: Path to Mfrac Model (../models/results/<Mfrac>.torch)"
    echo "4: Path to Baseline Model (results/<Baseline>.torch)"
    echo "5: Path to Pred Model (results/<Pred>.torch)"
    echo "6: Path to Truth Model (results/<Truth>.torch)"
    echo "7: Path to Output (results/<scores>.root)"
    exit 1
fi

in_file=$1
Efrac_model=$2
Mfrac_model=$3
in_baseline=$4
in_pred=$5
in_truth=$6
out_file=$7

rand=$RANDOM
nohup python -u eval_scores.py $in_file $Efrac_model $Mfrac_model $in_baseline $in_pred $in_truth $out_file > "eval_scores_$rand.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "eval_scores_$rand.log"
