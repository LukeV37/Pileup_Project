#!/bin/bash
out_dir=$1
mkdir -p $out_dir
nohup python3 -u preprocessing.py $out_dir > "${outdir}preprocessing.log" 2>&1 &
