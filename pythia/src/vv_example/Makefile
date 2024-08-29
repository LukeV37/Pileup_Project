CPP=g++
LD=g++
SPECIALFLAGS=-O2
ROOTCFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

CFLAGS = $(SPECIALFLAGS) -I.
LFLAGS = -L.

RCXX=$(CFLAGS) $(ROOTCFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

SRC=v.cpp v_dict.cpp

v_dict.cpp: LinkDef.h
	rootcling v_dict.cpp LinkDef.h

%.o: %.cpp
	$(CPP) $(RCXX) -c $<

all: $(SRC:.cpp=.o)
	$(LD) $(SRC:.cpp=.o) $(RLXX) -o v.exe

clean:
	rm -f *~ v_dict* *.o *.exe
