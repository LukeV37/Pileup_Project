#/bin/bash
cd ../Pythia-8.312
./configure --with-root --with-hepmc2=../HepMC-2.06.11/hepmc-install --with-fastjet3=../FastJet-3.4.2/fastjet-install --prefix=$PWD --with-gzip
make -j8
