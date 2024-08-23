#!/bin/bash

cd src
make generate_root
./run_root $1 $2 $3 $4
make clean
cd ..

name="dataset_$3_mu$2_NumEvents$1_MinJetpT$4.root"
echo "Processing: $name"
mv output/$name ./output/tmp.root
cd src
root -q -l -b add_true_pufr.C
root -q -l -b add_ttbar_match.C
cd ..
mv ./output/tmp.root output/$name
