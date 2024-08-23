#!/bin/bash
cd ../pythia
./generate_hepmc.sh
cd ../delphes
./convert_pileup.sh
./ATLAS_sim.sh
echo "DONE. If no errors, then the dataset has been generated!"
