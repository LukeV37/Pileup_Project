#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

int main()
{
	Pythia pythia;
	pythia.readFile("ttbar.cmnd");
    Pythia8ToHepMC HStoHepMC("../output/ttbar.hepmc");

	int nEventsHS = pythia.mode("Main:numberOfEvents");
  	int nAbortHS = pythia.mode("Main:timesAllowErrors");

	if (!pythia.init()) return 1;

	int iAbortHS = 0;
	for(int i=0;i<nEventsHS;i++){
    		if (!pythia.next()) {
      			if (++iAbortHS < nAbortHS) continue;
			    std::cout << " Event generation aborted prematurely, owing to error!\n";
      			break;
    	    }
        HStoHepMC.writeNextEvent( pythia );
    }

	Pythia pythiaPU;
	pythiaPU.readFile("pileup.cmnd");
    Pythia8ToHepMC PUtoHepMC("../output/pileup.hepmc");

	int nEventsPU = pythiaPU.mode("Main:numberOfEvents");
  	int nAbortPU = pythiaPU.mode("Main:timesAllowErrors");

	if (!pythiaPU.init()) return 1;

	int iAbortPU = 0;
	for(int i=0;i<nEventsPU;i++){
    		if (!pythiaPU.next()) {
      			if (++iAbortPU < nAbortPU) continue;
			    std::cout << " Event generation aborted prematurely, owing to error!\n";
      			break;
    	    }
        PUtoHepMC.writeNextEvent( pythiaPU );
    }

	return 0;
}
