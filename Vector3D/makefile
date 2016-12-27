# makefile
# Purpose: Create the executable for the main.cpp file to test Vector3D class.

CC = g++
CFLAGS = -I.
DEPS = Vector3D.h
OBJ = main.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)
