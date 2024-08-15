#!/bin/bash
dir=$PWD
rm -f output/dataset.root
cd ../software/Delphes-3.5.0
./DelphesHepMC2 ../../delphes/okstate_card_ATLAS_PileUp.tcl ../../delphes/output/dataset.root ../../pythia/output/ttbar.hepmc
./DelphesHepMC2 delphes_card_ATLAS_PileUp.tcl ../../delphes/output/dataset.root ../../pythia/output/ttbar.hepmc
cd $dir
#../software/Delphes-3.5.0/hepmc2pileup TEST.PILEUP ../pythia/output/pileup.hepmc
