#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 3 agruments"
    echo "1: Process {diHiggs|4b}"
    echo "2: Average PU, mu, (int)"
    echo "3: Num Events from MadGraph (int)"
    echo "4: MinJetpT (float)"
    exit 1
fi

cd src
make generate_dihiggs
./run_dihiggs $1 $2 $3 $4
make clean
cd ..

name="dataset_$1_mu$2_NumEvents$3_MinJetpT$4.root"
echo "Processing: $name"
cd src/scripts
root -q -l -b add_JVT.C\(\""$name"\"\)
root -q -l -b add_true_pufr.C\(\""$name"\"\)
#root -q -l -b add_ttbar_match.C\(\""$name"\"\)
cd ../..
