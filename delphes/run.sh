#!/bin/bash
cd ../pythia
./run.sh hepmc
cd ../delphes
./convert_pileup.sh
./ATLAS_sim.sh
echo "DONE. If no errors, then the dataset has been generated!"
