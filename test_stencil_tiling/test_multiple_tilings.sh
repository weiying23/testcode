#!/bin/bash

# Test various tiling configurations to find optimal size
echo "Testing different tiling configurations on Apple M4 chip"
echo "=========================================================="
echo ""

# Define tiling configurations to test
# Format: TILE_X TILE_Y TILE_Z
configs=(
    "8 8 256"
    "16 16 256"
    "32 32 256"
    "64 64 256"
    "16 16 128"
    "16 16 64"
    "16 16 32"
    "32 32 128"
    "32 32 64"
    "32 32 32"
    "64 32 64"
    "64 64 64"
)

best_time=999999
best_config=""

for config in "${configs[@]}"; do
    read TILE_X TILE_Y TILE_Z <<< "$config"
    echo "Testing TILE_X=$TILE_X, TILE_Y=$TILE_Y, TILE_Z=$TILE_Z"
    
    # Modify the source file
    sed -i.bak "s/#define TILE_X [0-9]*/#define TILE_X $TILE_X/" test_tiling.c
    sed -i.bak "s/#define TILE_Y [0-9]*/#define TILE_Y $TILE_Y/" test_tiling.c
    sed -i.bak "s/#define TILE_Z [0-9]*/#define TILE_Z $TILE_Z/" test_tiling.c
    
    # Compile
    clang -O3 -Xpreprocessor -pthread test_tiling.c -o test_tiling 2>&1 | grep -i error
    
    if [ $? -eq 0 ]; then
        echo "Compilation failed!"
        continue
    fi
    
    # Run and extract optimized tiling time
    output=$(./test_tiling 2>&1)
    time=$(echo "$output" | grep "Optimized tiling version:" | awk '{print $4}')
    
    if [ ! -z "$time" ]; then
        echo "  -> Time: $time seconds"
        
        # Compare times (using bc for floating point)
        if (( $(echo "$time < $best_time" | bc -l) )); then
            best_time=$time
            best_config="$config"
        fi
    else
        echo "  -> Failed to extract time"
    fi
    echo ""
done

echo "=========================================================="
echo "Best configuration: $best_config"
echo "Best time: $best_time seconds"
echo ""

# Restore best configuration
read TILE_X TILE_Y TILE_Z <<< "$best_config"
sed -i.bak "s/#define TILE_X [0-9]*/#define TILE_X $TILE_X/" test_tiling.c
sed -i.bak "s/#define TILE_Y [0-9]*/#define TILE_Y $TILE_Y/" test_tiling.c
sed -i.bak "s/#define TILE_Z [0-9]*/#define TILE_Z $TILE_Z/" test_tiling.c

# Final compilation with best config
clang -O3 -Xpreprocessor -pthread test_tiling.c -o test_tiling
echo "Recompiled with best configuration"
