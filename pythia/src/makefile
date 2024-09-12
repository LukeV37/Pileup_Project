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

SRC1=generate_ttbar.cpp generate_ttbar_dict.cpp
SRC2=generate_dihiggs.cpp generate_dihiggs_dict.cpp

generate_ttbar_dict.cpp: ./include/LinkDef.h
	@rootcling generate_ttbar_dict.cpp ./include/LinkDef.h

generate_dihiggs_dict.cpp: ./include/LinkDef.h
	@rootcling generate_dihiggs_dict.cpp ./include/LinkDef.h

%.o: %.cpp
	@$(CPP) $(RCXX) -c $< -w -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

generate_ttbar: $(SRC1:.cpp=.o)
	@$(LD) $(SRC1:.cpp=.o) $(RLXX) -o run_ttbar -w -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

generate_dihiggs: $(SRC2:.cpp=.o)
	@$(LD) $(SRC2:.cpp=.o) $(RLXX) -o run_dihiggs -w  -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2 -DGZIP -lz -L../lib -Wl,-rpath,../lib

generate_hepmc: generate_hepmc.cpp
	@g++ generate_hepmc.cpp -o run_hepmc -w  -I $(SOFTWARE_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SOFTWARE_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SOFTWARE_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/include -L$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/HepMC-2.06.11/hepmc-install/lib -lHepMC -DHEPMC2

clean:
	@rm -f *~ generate_*_dict* *.o *.exe
	@rm -f run_ttbar
	@rm -f run_dihiggs
	@rm -f run_hepmc
