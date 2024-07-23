SOFTWARE_PATH=/usr/local

generate_dataset: generate_dataset.cc
	g++ generate_dataset.cc -o run_dataset -w  -I $(SOFTWARE_PATH)/pythia8312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread  -L $(SOFTWARE_PATH)/pythia8312/lib -Wl,-rpath,$(SOFTWARE_PATH)/pythia8312/lib -lpythia8 -ldl -I$(SOFTWARE_PATH)/fastjet-install/include -L$(SOFTWARE_PATH)/fastjet-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/fastjet-install/lib -lfastjet -L$(SOFTWARE_PATH)/rootpy27/root_install/lib -Wl,-rpath,$(SOFTWARE_PATH)/rootpy27/root_install/lib -lCore -pthread -std=c++17 -m64 -fPIC -I$(SOFTWARE_PATH)/rootpy27/root_install/include -L$(SOFTWARE_PATH)/rootpy27/root_install/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -Wl,-rpath,$(SOFTWARE_PATH)/rootpy27/root_install/lib -pthread -lm -ldl -rdynamic -DPY8ROOT -I$(SOFTWARE_PATH)/hepmc2/hepmc2-install/include -L$(SOFTWARE_PATH)/hepmc2/hepmc2-install/lib -Wl,-rpath,$(SOFTWARE_PATH)/hepmc2/hepmc2-install/lib -lHepMC -DHEPMC2

clean:
	@rm -f run_dataset
