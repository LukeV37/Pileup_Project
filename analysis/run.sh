#!/bin/bash
for entry in ntuples/*
do
    echo "Processing: $entry"
    mv $entry ./ntuples/dataset.root
    cd src
    root -q -l -b add_true_pufr.C
    root -q -l -b add_ttbar_match.C
    cd ..
    mv ./ntuples/dataset.root $entry
    echo
done
