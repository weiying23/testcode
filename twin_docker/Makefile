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