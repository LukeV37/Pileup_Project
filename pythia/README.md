## Quick Start
To run a pythia simulation use either of the following commands:
```
# Must pass 4 arguments of type int, int, string, float
./generate_root.sh {Num HS Events} {Average Pileup mu} {ttbar|zprime} {Min Jet pT (GeV)}

# No arguments needed. Configured by .cmnd files
./generate_hepmc.sh
```
These scripts compile and run the code found in the `src` folder. 

The output can be found in the `output` folder.

To modify the pythia simulation settings, please modify the `.cmnd` files found in `src` folder.

## ROOT Output
The script `./generate_root.sh` first makes the source code in `src` then calls the binary `./run_root`. This binary is passed multiple arguments in the following order: number of hard scatter events, number of average number of pileup interactions per event $\mu$, hard scatter process {ttbar|zprime}, and minimum jet pT in GeV used by fastjet.

For example:
```
cd src
make generate_root
# Generate 100 ttbar events with $\mu=60$ and min jet pT of 25GeV
./run_root 100 60 ttbar 25
make clean
```

To edit the number of events or other parameters of the simluation, modify the code in the `src` folder accordingly. 

## HepMC Output
The script `./generate_hepmc.sh` first makes the source code in `src` then calls binary `./run_hepmc`. The settings are directly read from the `.cmnd` files. The output can be found in the `output` folder.

## Validation Plots
Jupyter notebook validation plots can be seen in the `plots` folder.
