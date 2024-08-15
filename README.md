Clone the repo over ssh using:
```
git clone --recursive git@github.com:LukeV37/Pileup_Dataset_Production.git
```

Install the submodules:
```
./install.sh
```

Please be patient while submodules build...

To run the chain Pythia Sim→HepMC Output→Delphes Sim→ROOT Output:
```
./run.sh
```

The `pythia` folder contains code to run a simulation to generate particles. See `pythia/README.md` for more details.

The `delphes` folder contains code to simulate the ATLAS detector. See `delphes/README.md` for more details.

Tested on Ubuntu 22.04 machine and on EL9 lxplus machine.
