## Quick Start
To run a delphes simulation, please run the following command:
```
./run.sh
```
This will read the hepmc files from the `../pythia/output` directory, convert pileup, and run ATLAS style simulation. The results of the simulation are stored in `dataset.root` in the `output` folder.

## How to Modify Configuration
To modify configuration of the delphes simulation, please modify `okstate_card_ATLAS_PileUp.tcl`.

The average number of pileup can be changed by editing the following
```
set MeanPileUp 60
```

More delphes documentation can be found [here](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook).
