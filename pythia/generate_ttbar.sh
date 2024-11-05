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
make generate_ttbar
./run_ttbar $1 $2 $3 $4
make clean
cd ..

name="dataset_$3_mu$2_NumEvents$1_MinJetpT$4.root"
echo "Processing: $name"
cd src/scripts
root -q -l -b add_JVT.cpp\(\""$name"\"\)
root -q -l -b add_true_pufr.cpp\(\""$name"\"\)
root -q -l -b add_ttbar_match.cpp\(\""$name"\"\)
#make
#make all
#./add_Likelihood.exe $name
#make clean
root -q -l -b add_true_EMfrac.cpp\(\""$name"\"\)
cd ../..
