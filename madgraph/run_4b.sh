#!/bin/bash
cd output
../../software/MadGraph5-v3.5.5/bin/mg5_aMC ../process_cards/4b_proc_card.dat

# Edit Run Card
Beam_Energy=7000.0
num_Events=20000
min_ptb_filter=60 #GeV

# Apply changes to run card
sed -i "s/6500.0/$Beam_Energy/" 4b/Cards/run_card.dat
sed -i "s/\(.*\)= nevents\(.*\)/ $num_Events = nevents\2/" 4b/Cards/run_card.dat
sed -i "s/0.0  = ptb/$min_ptb_filter  = ptb/" 4b/Cards/run_card.dat

# Generate LHE File
4b/bin/generate_events

# Clean up workspace
rm py.py
