Before delphes simluation is run, pileup must be converted from hepmc2 to a bitstream using the following:
```
./convert_pileup.sh
```
This will create a `pythia.pileup` in the `output` folder.

To run the delphes simluation use:
```
./run_ATLAS_sim.sh
```
This will create a `dataset.root` in the `output` folder.

To modify configuration of the delphes simulation, please modify `okstate_card_ATLAS_PileUp.tcl`.
