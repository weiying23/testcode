#!/bin/bash
# test_dist.sh - Test distributed s-step GMRES

set -e

echo "=============================================="
echo "Distributed s-step GMRES Tests"
echo "=============================================="

# Compile
echo "Compiling..."
mpicxx -std=c++11 -O3 -o sstep_gmres_dist sstep_gmres_dist.cpp

echo ""
echo "=== Phase 1: Single process validation ==="
echo "Test: n=400, s=3, m=15, np=1"
mpirun -np 1 ./sstep_gmres_dist 400 3 15 0 1e-8

echo ""
echo "=== Phase 2: Multi-process correctness ==="
echo "Test: n=400, s=3, m=15, np=4"
mpirun -np 4 ./sstep_gmres_dist 400 3 15 0 1e-8

echo ""
echo "=== Phase 3: Large-scale performance ==="
echo "Test: n=100000, s=3, m=30, np=10"
mpirun -np 10 ./sstep_gmres_dist 100000 3 30 0 1e-8

echo ""
echo "=== Comparison with redundant storage version ==="
echo "Redundant version (np=10, n=100000):"
mpirun -np 10 ./sstep_gmres_paper 100000 3 30 0 1e-8 2>&1 | grep -E "Time|Converged|Global" || echo "Redundant version not available"

echo ""
echo "Tests complete!"