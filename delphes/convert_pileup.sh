#!/bin/bash
dir=$PWD
cd ../software/Delphes-3.5.0
./hepmc2pileup ../../delphes/output/pythia.pileup ../../pythia/output/pileup.hepmc
cd $dir
