#!/bin/bash
../software/MadGraph5-v3.5.5/bin/mg5_aMC proc_card.dat
epspdf /tmp/diagrams_*hh.eps diagrams.pdf

# Edit Run Card
Beam_Energy=7000.0
sed -i "s/6500.0/$Beam_Energy/" DiHiggs/Cards/run_card.dat
num_Events=10000
sed -i "s/\(.*\)= nevents\(.*\)/ $num_Events = nevents\2/" DiHiggs/Cards/run_card.dat

# Generate LHE File
./DiHiggs/bin/generate_events

rm py.py
