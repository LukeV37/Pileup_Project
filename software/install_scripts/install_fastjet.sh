#!/bin/bash
cd ../FastJet-3.4.2
mkdir fastjet-install
./autogen.sh --prefix="$PWD/fastjet-install/"
make -j4
make install
