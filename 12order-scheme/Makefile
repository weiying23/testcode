# Detect OS
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    # macOS settings
    CC = clang
    CFLAGS_COMMON = -Wall -Wextra -O3 -mcpu=apple-m1 -funroll-loops -ffast-math
    OMP_FLAGS = -Xpreprocessor -fopenmp -lomp
else
    # Linux settings
    CC = gcc
    CFLAGS_COMMON = -Wall -Wextra -O3 -march=native -funroll-loops -ffast-math
    OMP_FLAGS = -fopenmp
endif

CFLAGS_DEBUG = -g -O0 -DDEBUG

# Original version
CFLAGS_ORIG = $(CFLAGS_COMMON)

# Optimized version
CFLAGS_OPT = $(CFLAGS_COMMON)

all: original optimized split
	@echo "Built successfully."
	@echo "Note: OpenMP version requires libomp (brew install libomp on macOS)"

original: 3d-2nd-deriv-12th-original.c
	$(CC) $(CFLAGS_ORIG) -o $@ $< -lm
	@echo "Built original version"

optimized: 3d-2nd-deriv-12th-optimized.c
	$(CC) $(CFLAGS_OPT) -o $@ $< -lm
	@echo "Built optimized version (fused)"

split: 3d-2nd-deriv-12th-split.c
	$(CC) $(CFLAGS_OPT) -o $@ $< -lm
	@echo "Built split version (3x 1D passes)"

# OpenMP version (optional, requires libomp)
# On macOS: brew install libomp, then run 'make omp'
omp: 3d-2nd-deriv-12th-omp.c
	$(CC) $(CFLAGS_OPT) $(OMP_FLAGS) -o $@ $< -lm
	@echo "Built OpenMP multi-threaded version"

debug: 3d-2nd-deriv-12th-original.c
	$(CC) $(CFLAGS_DEBUG) -o $@ $< -lm

compare: original optimized split
	@echo "========================================"
	@echo "Running ORIGINAL version:"
	@echo "========================================"
	./original
	@echo ""
	@echo "========================================"
	@echo "Running OPTIMIZED (fused) version:"
	@echo "========================================"
	./optimized
	@echo ""
	@echo "========================================"
	@echo "Running SPLIT (3x 1D passes) version:"
	@echo "========================================"
	./split

clean:
	rm -f original optimized split omp

.PHONY: all original optimized omp debug compare clean
