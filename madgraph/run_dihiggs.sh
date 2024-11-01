#!/bin/bash
cd output
../../software/MadGraph5-v3.5.5/bin/mg5_aMC ../process_cards/DiHiggs_proc_card.dat

# Edit Run Card
Beam_Energy=7000.0
num_Events=20000
seed=1234

# Apply changes to run card
sed -i "s/6500.0/$Beam_Energy/" DiHiggs/Cards/run_card.dat
sed -i "s/\(.*\)= nevents\(.*\)/ $num_Events = nevents\2/" DiHiggs/Cards/run_card.dat
sed -i "s/\(.*\)= iseed\(.*\)/ $seed = iseed\2/" DiHiggs/Cards/run_card.dat

# Generate LHE File
DiHiggs/bin/generate_events

# Clean up workspace
rm py.py
