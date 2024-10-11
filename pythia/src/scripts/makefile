SOFTWARE_PATH=../../software/

cppPP=g++
LD=g++
SPEcppIALFLAGS=-O2
ROOTcppFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

cppFLAGS = $(SPEcppIALFLAGS) -I.
LFLAGS = -L.

RcppXX=$(cppFLAGS) $(ROOTcppFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

SRcpp=add_Likelihood.cpp add_Likelihood_dict.cpp

add_Likelihood_dict.cpp: ../include/LinkDef.h
	@rootcling add_Likelihood_dict.cpp ../include/LinkDef.h

%.o: %.cpp
	@$(cppPP) $(RcppXX) -c $< 

all: $(SRcpp:.cpp=.o)
	@$(LD) $(SRcpp:.cpp=.o) $(RLXX) -o add_Likelihood.exe

clean:
	@rm -f *~ add*_dict* *.o *.exe
