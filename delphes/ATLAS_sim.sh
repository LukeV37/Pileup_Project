#!/bin/bash
dir=$PWD
rm -f output/dataset.root
cd ../software/Delphes-3.5.0
./DelphesHepMC2 ../../delphes/okstate_card_ATLAS_PileUp.config ../../delphes/output/dataset.root ../../pythia/output/ttbar.hepmc
#./DelphesHepMC2 delphes_card_ATLAS_PileUp.config ../../delphes/output/dataset.root ../../pythia/output/ttbar.hepmc
cd $dir
