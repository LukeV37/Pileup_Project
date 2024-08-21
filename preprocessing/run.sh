#!/bin/bash
for entry in input/*
do
    echo "Processing: $entry"
    mv $entry ./input/dataset.root
    cd src
    root -q -l -b add_true_pufr.C
    root -q -l -b add_ttbar_match.C
    cd ..
    mv ./input/dataset.root $entry
    echo
done
