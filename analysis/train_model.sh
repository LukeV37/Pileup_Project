#!/bin/bash

if [ -z "$8" ]; then
    echo "Must enter 8 agruments"
    echo "1: Num Epochs e.g. 40"
    echo "2: Path to Input File (data/<file>.pkl)"
    echo "3: Path to Efrac Model (results/<Efrac>.torch)"
    echo "4: Path to Mfrac Model (results/<Mfrac>.torch)"
    echo "5: Path to Baseline Output (results/<baseline>.torch)"
    echo "6: Path to Truth Output (results/<truth>.torch)"
    echo "7: Path to Pred Output (results/<pred>.torch)"
    echo "8: Path to Output Directory (plots/<Dir Name>) e.g. plots/regression"
    exit 1
fi

epochs=$1
in_file=$2
Efrac_model=$3
Mfrac_model=$4
out_baseline=$5
out_truth=$6
out_pred=$7
out_dir=$8

mkdir -p $out_dir
nohup python -u DiHiggs_4b_Classifier.py $epochs $in_file $Efrac_model $Mfrac_model $out_baseline $out_truth $out_pred $out_dir > "${out_dir}/training.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "${out_dir}/training.log"
