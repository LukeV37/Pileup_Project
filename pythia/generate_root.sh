#!/bin/bash

cd src
make generate_root
#./run root num_events mu process minjetpT
./run_root 100 60 ttbar 25
./run_root 100 60 zprime 25
#./run_root 10000 200 ttbar 25
#./run_root 10000 200 zprime 25
make clean
cd ..

for entry in output/*.root
do
    echo "Processing: $entry"
    mv $entry ./output/tmp.root
    cd src
    root -q -l -b add_true_pufr.C
    root -q -l -b add_ttbar_match.C
    cd ..
    mv ./output/tmp.root $entry
    echo
done
