# Makefile for Tachyon CUDA example programs.
#
# Usage:
#   make all        - build all examples
#   make clean      - remove binaries
#   make 01_matmul_naive   - build a single example
#
# Override architecture:
#   make all ARCH="-arch=sm_80"

NVCC  ?= $(shell which nvcc 2>/dev/null || echo nvcc)
FLAGS  = -O2 -lineinfo
ARCH  ?= -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_89,code=sm_89 \
          -gencode arch=compute_90,code=sm_90

SRCS   = $(wildcard *.cu)
BINS   = $(SRCS:.cu=)

.PHONY: all clean

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $<

clean:
	rm -f $(BINS)
