#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 4 agruments"
    echo "1: Run Type {Efrac|Mfrac}"
    echo "2: Path to Input File (../pythia/output/<file>.root)"
    echo "3: Path to Output File (data/<file>.pkl)"
    echo "4: Path to Output Directory (plots/<Dir Name>) e.g. plots/preprocessing"
    exit 1
fi

run_type=$1
in_file=$2
out_file=$3
out_plots=$4

mkdir -p $out_plots
nohup python -u preprocessing.py $run_type $in_file $out_file $out_plots > "${out_plots}/preprocessing.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "${out_plots}/preprocessing.log"
