#!/bin/bash

if [ -z "$5" ]; then
    echo "Must enter 5 agruments"
    echo "1: Run Type {Efrac|Mfrac}"
    echo "2: Path to Signal File (../pythia/output/<sig>.root)"
    echo "3: Path to Background File (../pythia/output/<bkg>.root)"
    echo "4: Path to Output File (data/<file>.pkl)"
    echo "5: Path to Output Directory (plots/<Dir Name>) e.g. plots/preprocessing"
    exit 1
fi

run_type=$1
sig_file=$2
bkg_file=$3
out_file=$4
out_plots=$5

mkdir -p $out_plots
nohup python -u preprocessing_combined.py $run_type $sig_file $bkg_file $out_file $out_plots > "${out_plots}/preprocessing.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "${out_plots}/preprocessing.log"
