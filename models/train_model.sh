#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 4 agruments"
    echo "1: Num Epochs e.g. 40"
    echo "2: Path to Input File (data/<file>.pkl)"
    echo "3: Path to Output File (results/<file>.torch)"
    echo "4: Path to Output Directory (plots/<Dir Name>) e.g. plots/regression"
    exit 1
fi

epochs=$1
in_file=$2
out_file=$3
out_dir=$4
mkdir -p $out_dir
nohup python -u Jet_Attention_Model.py $epochs $in_file $out_file $out_dir > "${out_dir}/training.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "${out_dir}/training.log"
