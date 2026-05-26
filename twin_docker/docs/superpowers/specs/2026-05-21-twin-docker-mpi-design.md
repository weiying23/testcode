---
name: twin-docker-mpi-communication
description: Two Docker containers with MPI communication and data verification
metadata:
  type: project
---

# Twin Docker Container MPI Communication Design

## Overview

Two Docker containers communicate via MPI (Send/Recv), with data verification to ensure transmission correctness.

## Architecture

```
Docker Network (mpi_net)
├── Container 0 (172.19.0.2)
│   ├── OpenMPI + SSH
│   └─ MPI process (rank 0)
│
└── Container 1 (172.19.0.3)
    ├── OpenMPI + SSH
    └─ MPI process (rank 1)

Communication: MPI Send/Recv via Docker internal network
```

## Components

### Dockerfile

- Ubuntu 22.04 + OpenMPI + SSH
- Passwordless SSH configured for container-to-container access
- MPI program compiled at build time

### mpi_app.c

MPI program with Send/Recv and verification:
- Rank 0 sends data to rank 1
- Rank 1 verifies and sends confirmation
- Both ranks verify data matches theoretical pattern

### run_mpi.sh

Launcher script:
1. Create Docker network
2. Start two containers
3. Exchange SSH keys between containers
4. Configure SSH for passwordless access
5. Run MPI from container 0 (coordinator)
6. Cleanup containers and network

### cleanup.sh

Cleanup script - removes containers and optionally the image.

## Usage

```bash
# Build
make build

# Run (default 64KB)
./run_mpi.sh

# Run with custom size
./run_mpi.sh 102400

# Cleanup
./cleanup.sh --remove-image
```

## Communication Flow

```
Rank 0                         Rank 1
  │                              │
  │── MPI_Send(64KB) ────────────│── MPI_Recv
  │                              │── verify
  │                              │── MPI_Send(confirm)
  │── MPI_Recv ───────────────────│
  │── verify                     │
  │                              │
  └── PASSED                     └── PASSED
```

## Isolation

- Containers have independent MPI environment
- Host MPI completely unaffected
- Deleting containers/image removes all container files