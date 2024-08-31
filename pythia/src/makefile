SOFTWARE_PATH=../../software/

CPP=g++
LD=g++
SPECIALFLAGS=-O2
ROOTCFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

CFLAGS = $(SPECIALFLAGS) -I.
LFLAGS = -L.

RCXX=$(CFLAGS) $(ROOTCFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

SRC=generate_root.cpp generate_root_dict.cpp

generate_root_dict.cpp: ./include/LinkDef.h
	@rootcling generate_root_dict.cpp ./include/LinkDef.h

%.o: %.cpp
	@$(CPP) $(RCXX) -c $< -w -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2
 
generate_root: $(SRC:.cpp=.o)
	@$(LD) $(SRC:.cpp=.o) $(RLXX) -o run_root -w -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

generate_hepmc: generate_hepmc.cpp
	@g++ generate_hepmc.cpp -o run_hepmc -w  -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

clean:
	@rm -f *~ generate_root_dict* *.o *.exe
	@rm -f run_root
	@rm -f run_hepmc
