#!/bin/bash
cd pythia
./run.sh hepmc
cd ../delphes
./convert_pileup.sh
./run_ATLAS_sim.sh
echo "DONE. Dataset has been generated!"
