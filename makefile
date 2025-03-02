# Compiler
CC = g++

# Compiler flags
CFLAGS = -g

# Source files
SRCS = $(wildcard *.cc) 

# Object files
OBJS = $(SRCS:.cc=.o) 

# Executable name
EXEC = thesis_project

# Default target
all: $(EXEC)

# Link object files to create executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cc 
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC)

.PHONY: all clean