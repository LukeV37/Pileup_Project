Clone the repo over ssh using:
```
git clone --recursive git@github.com:LukeV37/Pileup_Dataset_Production.git
```

Install the submodules:
```
cd software/install_scipts
./install_submodules.sh
```

Please be patient while submodules build...

The `pythia` folder contains code to run a simulation to generate particles. See `pythia/README.md` for more details.

The `delphes` folder contains code to simulate the ATLAS detector. See `delphes/README.md` for more details.
