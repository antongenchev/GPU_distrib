# Makefile

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O2

# Output executable name
TARGET = program

# Source file
SRC = main.cu

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET)
