#!/bin/bash
cd ../MadGraph5-v3.5.5
wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.5.tar.gz
tar -xvzf MG5_aMC_v3.5.5.tar.gz
mv MG5_aMC_v3_5_5/* .
rm -r MG5_aMC_v3_5_5
rm MG5_aMC_v3.5.5.tar.*
