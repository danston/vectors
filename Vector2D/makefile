# makefile
# Purpose: Create the executable for the main.cpp file to test Vector2D class.

CC = g++
CFLAGS = -I.
DEPS = Vector2D.h
OBJ = main.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)
