# Compiler
NVCC = nvcc

# Compiler Flags
ARCH = -arch=sm_75
RDC = -rdc=true
CFLAGS = $(ARCH) $(RDC)

# Target
TARGET = levenshtein

# Source Files
SRCS = levenshtein_warp_shuffle.cu

# Rules
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)