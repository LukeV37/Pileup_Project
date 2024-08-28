## Quick Start
Clone the repo over ssh using:
```
git clone --recursive git@github.com:LukeV37/Pileup_Project.git
```

Or clone the repo over https using:
```
git clone --recursive https://github.com/LukeV37/Pileup_Project.git
```

Install the submodules:
```
./install.sh
```

Please be patient while submodules build...

See `software/README.md` for more information regarding submodules.


## How To Generate Datasets

### Pythia Datasets
To run Pythia8 simulation, run the following
```
cd pythia
./generate_root.sh
```
The root file generated by the pythia simulation will be found in `pythia/output`. 

For more information on pythia, please see `pythia/README.md`.

### Delphes Datasets
To run Delphes simulation, run the following
```
cd delphes
./run.sh
```
The root file generated by the delphes simluation will be found in `delphes/output`. 

For more information on delphes, please see `delphes/README.md`.

## Documentation
The `software` folder contains libraries needed for simulating the datasets. See `software/README.md` for more details.

The `pythia` folder contains code to run a simulation to generate particles. See `pythia/README.md` for more details.

The `delphes` folder contains code to simulate the ATLAS detector. See `delphes/README.md` for more details.

The `python` folder contains code that preprocesses the datasets and code for PyTorch model. See `python/README.md` for more details.

The `analysis` folder contains code used for physics analysis. See `analysis/README.md` for more details.


## Dependencies
Tested on Ubuntu 22.04, EL9 lxplus machine, and even WSL2.

Required Dependencies:
<ul>
<li>ROOTv6</li>
<li>autoconf</li>
<li>libtool</li>
<li>automake</li>
<li>tcl</li>
</ul>

ROOTv6 can be installed following instructions [here](https://root.cern/install/).

Other packages can be intalled with `sudo apt install {package}`

## Misc.

>[!WARNING]
> [ROOT](https://root.cern/install/) must be installed on the system before `./install.sh` script can be run. \
> Use `root-config --cflags --libs` to see if you have a successful ROOT install.

>[!NOTE]
> On GPURIG2, please add the following lines to your `.bashrc` file in your home directory:
> ```bash
> export ROOTSYS=/usr/local/rootpy27/root_install
> export PATH=$ROOTSYS/bin:$PATH
> ```
