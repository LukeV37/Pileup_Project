#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 4 agruments"
    echo "1: Num Events (int)"
    echo "2: Average PU, mu, (int)"
    echo "3: Process {ttbar|zprime}"
    echo "4: MinJetpT (float)"
    exit 1
fi

cd src
make generate_root
./run_root $1 $2 $3 $4
make clean
cd ..

name="dataset_$3_mu$2_NumEvents$1_MinJetpT$4.root"
echo "Processing: $name"
cd src
root -q -l -b add_JVT.C\(\""$name"\"\)
root -q -l -b add_true_pufr.C\(\""$name"\"\)
root -q -l -b add_ttbar_match.C\(\""$name"\"\)
cd ..
