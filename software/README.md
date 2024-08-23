## Software

### Pythia-8.312
[Pythia](https://pythia.org/) is an open source program for the generation of high-energy physics collisions and is used to simulate datasets. Please see `Pythia-8.312/examples` for tutorials. Most notably, take a look into `main142.cc` and `main132.cc`. To compile and run these examples, please first build pythia using `./configure` then `make` and use the following commands:
```
./runmains --run=###
./main###
```
Take a look at the output, then run `make clean` to delete and cleanup the directory.

Pythia must be configured with external libraries using the `./configure` script. Use a text editor to view the `./configure` script to see all supported options.

### FastJet-3.4.2
FastJet is open source software for jet finding (ArXiv:[1111.6097](https://arxiv.org/abs/1111.6097)). Pythia simulates and showers particle decays, and FastJet is used to group these particles into collimated streams of particles called jets. A popular jet finding algorithm is anti- $k_t$ (ArXiv:[0802.1189](https://arxiv.org/abs/0802.1189)). To install FastJet, please see `install_scripts/install_fastjet.sh`.

### Delphes-3.5.0
[Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes) is an open source project used for fast simulations of a generic collider experiment. Instead of a full detector simulation, Delphes works by applying various parameterizations to the *theoretical* output of pythia to make the data look more *experimental*. For more information on how to use Delphes, please refer to the [Delphes Workbook](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook).

### HepMC-2.06.11
HepMC is a plain text event record used in high energy physics to store monte carlo information such as particle properties and kinematics. Although HepMC3 exists, there were problems with Delphes and HepMC3, so it was decided to use HepMC2.

