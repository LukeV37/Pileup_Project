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

SRC=add_Likelihood.C add_Likelihood_dict.C

add_Likelihood_dict.C: ../include/LinkDef.h
	@rootcling add_Likelihood_dict.C ../include/LinkDef.h

%.o: %.C
	@$(CPP) $(RCXX) -c $< 

all: $(SRC:.C=.o)
	@$(LD) $(SRC:.C=.o) $(RLXX) -o add_Likelihood.exe

clean:
	@rm -f *~ add*_dict* *.o *.exe
