#!/bin/bash

# Cleanup script for MPI Docker containers
# Safely removes containers and optionally the image
# Does NOT affect host MPI environment

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