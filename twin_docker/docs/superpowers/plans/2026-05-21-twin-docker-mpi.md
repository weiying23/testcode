# Twin Docker MPI Communication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a system where two Docker containers communicate via MPI with data verification.

**Architecture:** Host network mode for direct MPI communication between containers. Dockerfile builds image with OpenMPI. launcher.sh spawns containers via mpirun. mpi_app.c handles Send/Recv with verification.

**Tech Stack:** Docker, OpenMPI, C, Bash

---

## File Structure

```
twin_docker/
├── Dockerfile          # OpenMPI container image
├── mpi_app.c           # MPI communication program
├── launcher.sh         # mpirun launcher script
├── cleanup.sh          # Container cleanup script
├── Makefile            # Build automation
└── shared/             # Optional shared directory
```

---

### Task 1: Create Shared Directory Structure

**Files:**
- Create: `shared/` directory

- [ ] **Step 1: Create shared directory**

```bash
mkdir -p shared
```

Run: `mkdir -p shared`
Expected: Directory created successfully

- [ ] **Step 2: Commit**

```bash
git add shared/.gitkeep
git commit -m "feat: add shared directory structure"
```

---

### Task 2: Write MPI Application (mpi_app.c)

**Files:**
- Create: `mpi_app.c`

- [ ] **Step 1: Write mpi_app.c**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define DEFAULT_DATA_SIZE 65536  // 64KB default

// Generate theoretical data pattern for verification
void generate_theoretical_data(char *data, int size, int rank) {
    for (int i = 0; i < size; i++) {
        data[i] = (char)((rank + i) % 256);
    }
}

// Verify received data against theoretical value
int verify_data(char *received, char *expected, int size) {
    for (int i = 0; i < size; i++) {
        if (received[i] != expected[i]) {
            printf("Verification FAILED at index %d: expected %d, got %d\n",
                   i, (int)expected[i], (int)received[i]);
            return 0;
        }
    }
    printf("Verification PASSED: all %d bytes match\n", size);
    return 1;
}

int main(int argc, char **argv) {
    int rank, size;
    int data_size = DEFAULT_DATA_SIZE;
    char *send_data = NULL;
    char *recv_data = NULL;
    char *theoretical = NULL;
    MPI_Status status;

    // Parse data size from argument if provided
    if (argc > 1) {
        data_size = atoi(argv[1]);
        if (data_size <= 0) {
            data_size = DEFAULT_DATA_SIZE;
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        printf("Error: This program requires exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    printf("Process %d started, data size: %d bytes\n", rank, data_size);

    // Allocate buffers
    send_data = (char *)malloc(data_size);
    recv_data = (char *)malloc(data_size);
    theoretical = (char *)malloc(data_size);

    if (!send_data || !recv_data || !theoretical) {
        printf("Error: Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }

    // Generate send data (rank's pattern)
    generate_theoretical_data(send_data, data_size, rank);

    if (rank == 0) {
        // Rank 0 sends data to rank 1
        printf("Rank 0: Sending %d bytes to rank 1...\n", data_size);
        MPI_Send(send_data, data_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // Receive confirmation from rank 1
        printf("Rank 0: Waiting for confirmation from rank 1...\n");
        MPI_Recv(recv_data, data_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);

        // Verify confirmation data (should be rank 1's pattern)
        generate_theoretical_data(theoretical, data_size, 1);
        int result = verify_data(recv_data, theoretical, data_size);

        if (result) {
            printf("Rank 0: Communication complete and verified!\n");
        } else {
            printf("Rank 0: Verification failed!\n");
        }

    } else if (rank == 1) {
        // Rank 1 receives data from rank 0
        printf("Rank 1: Waiting for data from rank 0...\n");
        MPI_Recv(recv_data, data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        // Verify received data (should be rank 0's pattern)
        generate_theoretical_data(theoretical, data_size, 0);
        int result = verify_data(recv_data, theoretical, data_size);

        if (result) {
            printf("Rank 1: Received data verified!\n");
        } else {
            printf("Rank 1: Verification failed!\n");
        }

        // Send confirmation back to rank 1 (rank 1's pattern)
        printf("Rank 1: Sending confirmation to rank 0...\n");
        MPI_Send(send_data, data_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

        printf("Rank 1: Communication complete!\n");
    }

    // Cleanup
    free(send_data);
    free(recv_data);
    free(theoretical);

    MPI_Finalize();
    return 0;
}
```

Run: Create file at `mpi_app.c`
Expected: File created with MPI Send/Recv implementation

- [ ] **Step 2: Commit**

```bash
git add mpi_app.c
git commit -m "feat: add MPI communication program with verification"
```

---

### Task 3: Write Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM ubuntu:22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install OpenMPI and build tools
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set MPI environment
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Create app directory
WORKDIR /app

# Copy source and build
COPY mpi_app.c /app/mpi_app.c
RUN mpicc -o mpi_app mpi_app.c

# Set entrypoint
ENTRYPOINT ["mpirun", "--allow-run-as-root", "-np", "1", "/app/mpi_app"]
```

Run: Create file at `Dockerfile`
Expected: File created with OpenMPI installation

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile with OpenMPI environment"
```

---

### Task 4: Write Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write Makefile**

```makefile
# Twin Docker MPI Communication Makefile

IMAGE_NAME = mpi_app
CONTAINER_PREFIX = mpi_container

.PHONY: all build clean run cleanup

all: build

# Build Docker image
build:
	docker build -t $(IMAGE_NAME):latest .

# Build MPI app locally (for testing without Docker)
build-local:
	mpicc -o mpi_app mpi_app.c

# Run locally without Docker (for testing)
run-local:
	mpirun -np 2 ./mpi_app

# Run with Docker via launcher
run:
	mpirun -np 2 ./launcher.sh

# Cleanup containers and optionally image
cleanup:
	./cleanup.sh

# Full cleanup including image
clean-all:
	./cleanup.sh --remove-image

clean:
	rm -f mpi_app
```

Run: Create file at `Makefile`
Expected: File created with build/run/cleanup targets

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile for build automation"
```

---

### Task 5: Write Launcher Script

**Files:**
- Create: `launcher.sh`

- [ ] **Step 1: Write launcher.sh**

```bash
#!/bin/bash

# Launcher script for MPI Docker containers
# This script is executed by mpirun on the host
# It launches a Docker container with host network mode

# Get MPI rank from environment (set by mpirun)
RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}

# Container name based on rank
CONTAINER_NAME="mpi_container_${RANK}"

# Shared directory for optional file output
SHARED_DIR="$(pwd)/shared"

# Data size (can be overridden via environment)
DATA_SIZE=${MPI_DATA_SIZE:-65536}

echo "Launcher: Starting container for rank ${RANK}"

# Run Docker container with host network mode
# --network=host allows MPI to communicate directly
# --rm auto-removes container after exit
# Mount shared directory for debugging
docker run \
    --network=host \
    --rm \
    --name "${CONTAINER_NAME}" \
    -v "${SHARED_DIR}:/shared" \
    -e OMPI_COMM_WORLD_RANK=${RANK} \
    -e MPI_DATA_SIZE=${DATA_SIZE} \
    mpi_app:latest \
    ${DATA_SIZE}

echo "Launcher: Container for rank ${RANK} completed"
```

Run: Create file at `launcher.sh`
Expected: File created with Docker launch logic

- [ ] **Step 2: Make script executable**

```bash
chmod +x launcher.sh
```

Run: `chmod +x launcher.sh`
Expected: Script made executable

- [ ] **Step 3: Commit**

```bash
git add launcher.sh
git commit -m "feat: add launcher script for MPI Docker containers"
```

---

### Task 6: Write Cleanup Script

**Files:**
- Create: `cleanup.sh`

- [ ] **Step 1: Write cleanup.sh**

```bash
#!/bin/bash

# Cleanup script for MPI Docker containers
# Safely removes containers and optionally the image
# Does NOT affect host MPI installation

REMOVE_IMAGE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --remove-image) REMOVE_IMAGE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Cleanup: Stopping and removing MPI containers..."

# Remove containers (rank 0 and rank 1)
for i in 0 1; do
    CONTAINER_NAME="mpi_container_${i}"
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Cleanup: Removing container ${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    else
        echo "Cleanup: Container ${CONTAINER_NAME} not found (already removed)"
    fi
done

# Optionally remove the image
if [ "$REMOVE_IMAGE" = true ]; then
    echo "Cleanup: Removing MPI image..."
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^mpi_app:latest$"; then
        docker rmi mpi_app:latest 2>/dev/null || true
        echo "Cleanup: Image removed"
    else
        echo "Cleanup: Image not found"
    fi
fi

echo "Cleanup: Complete. Host MPI environment is unaffected."
echo "Cleanup: Host OpenMPI remains at system location."
```

Run: Create file at `cleanup.sh`
Expected: File created with container cleanup logic

- [ ] **Step 2: Make script executable**

```bash
chmod +x cleanup.sh
```

Run: `chmod +x cleanup.sh`
Expected: Script made executable

- [ ] **Step 3: Commit**

```bash
git add cleanup.sh
git commit -m "feat: add cleanup script for container removal"
```

---

### Task 7: Build and Test Docker Image

**Files:**
- None (testing phase)

- [ ] **Step 1: Build Docker image**

```bash
make build
```

Run: `make build`
Expected: Image `mpi_app:latest` created successfully

- [ ] **Step 2: Verify image exists**

```bash
docker images mpi_app:latest
```

Run: `docker images mpi_app:latest`
Expected: Image listed with correct tag

---

### Task 8: Test MPI Communication

**Files:**
- None (testing phase)

- [ ] **Step 1: Run MPI test**

```bash
mpirun -np 2 ./launcher.sh
```

Run: `mpirun -np 2 ./launcher.sh`
Expected: Two containers launch, communicate, and verification passes

- [ ] **Step 2: Check output for verification success**

Look for output:
```
Rank 0: Verification PASSED
Rank 1: Verification PASSED
```

---

### Task 9: Test Cleanup Script

**Files:**
- None (testing phase)

- [ ] **Step 1: Run cleanup**

```bash
./cleanup.sh
```

Run: `./cleanup.sh`
Expected: Containers removed, output confirms cleanup

- [ ] **Step 2: Verify containers are gone**

```bash
docker ps -a | grep mpi_container
```

Run: `docker ps -a | grep mpi_container`
Expected: No matching containers found

- [ ] **Step 3: Verify host MPI still works**

```bash
mpirun --version
```

Run: `mpirun --version`
Expected: Host mpirun still functional, version displayed

---

### Task 10: Final Commit and Documentation Update

**Files:**
- None (finalization)

- [ ] **Step 1: Add gitkeep for shared directory**

```bash
touch shared/.gitkeep
git add shared/.gitkeep
```

Run: `touch shared/.gitkeep && git add shared/.gitkeep`
Expected: Git tracks shared directory

- [ ] **Step 2: Final commit**

```bash
git status
git commit -m "feat: complete twin docker MPI communication system"
```

Run: Review and commit any remaining changes
Expected: All changes committed