# makefile
# Purpose: Create the executable for the main.cpp file to test VectorXD library.

CC = g++
CFLAGS = -I.
DEPS = VectorXD 
OBJ = main.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)
