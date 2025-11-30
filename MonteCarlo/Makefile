MPICXX = mpic++
CXXFLAGS = -Wall -O3

SRC = pi.cpp pi.h
EXEC = pi

all: $(EXEC)

$(EXEC): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXEC)
