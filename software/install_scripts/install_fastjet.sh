#!/bin/bash
cd ../FastJet-3.4.2
git submodule init
git submodule update
mkdir fastjet-install
./autogen.sh --prefix="$PWD/fastjet-install/"
make -j
make install
