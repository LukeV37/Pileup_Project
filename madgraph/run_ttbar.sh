#!/bin/bash
cd output
../../software/MadGraph5-v3.5.5/bin/mg5_aMC ../process_cards/ttbar_proc_card.dat

# Edit Run Card
Beam_Energy=7000.0
sed -i "s/6500.0/$Beam_Energy/" ttbar/Cards/run_card.dat
num_Events=10000
sed -i "s/\(.*\)= nevents\(.*\)/ $num_Events = nevents\2/" ttbar/Cards/run_card.dat
# Filter mininum pt of b quark
#sed -i "s/0.0  = ptb/60.0  = ptb/" ttbar/Cards/run_card.dat

# Generate LHE File
ttbar/bin/generate_events

rm py.py
