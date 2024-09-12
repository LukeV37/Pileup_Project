#!/bin/bash

if [ -z "$2" ]; then
    echo "Must enter 4 agruments"
    echo "1: Average PU, mu, (int)"
    echo "2: MinJetpT (float)"
    exit 1
fi

cd src
make generate_dihiggs
./run_dihiggs $1 $2
make clean
cd ..

name="dataset_diHiggs_4b_mu$1_NumEvents10k_MinJetpT$2.root"
echo "Processing: $name"
cd src/scripts
root -q -l -b add_JVT.C\(\""$name"\"\)
root -q -l -b add_true_pufr.C\(\""$name"\"\)
#root -q -l -b add_ttbar_match.C\(\""$name"\"\)
cd ../..
