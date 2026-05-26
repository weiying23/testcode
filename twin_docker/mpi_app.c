#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define DEFAULT_DATA_SIZE 65536

void generate_theoretical_data(char *data, int size, int rank) {
    for (int i = 0; i < size; i++) {
        data[i] = (char)((rank + i) % 256);
    }
}

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

    if (argc > 1) {
        data_size = atoi(argv[1]);
        if (data_size <= 0) data_size = DEFAULT_DATA_SIZE;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        printf("Error: This program requires exactly 2 processes (got %d)\n", size);
        MPI_Finalize();
        return 1;
    }

    printf("Process %d started, world size: %d, data size: %d bytes\n", rank, size, data_size);

    send_data = (char *)malloc(data_size);
    recv_data = (char *)malloc(data_size);
    theoretical = (char *)malloc(data_size);

    if (!send_data || !recv_data || !theoretical) {
        printf("Error: Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }

    generate_theoretical_data(send_data, data_size, rank);

    if (rank == 0) {
        printf("Rank 0: Sending %d bytes to rank 1...\n", data_size);
        MPI_Send(send_data, data_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        printf("Rank 0: Waiting for confirmation from rank 1...\n");
        MPI_Recv(recv_data, data_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);

        generate_theoretical_data(theoretical, data_size, 1);
        if (verify_data(recv_data, theoretical, data_size)) {
            printf("Rank 0: Communication complete and verified!\n");
        } else {
            printf("Rank 0: Verification failed!\n");
        }
    } else {
        printf("Rank 1: Waiting for data from rank 0...\n");
        MPI_Recv(recv_data, data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        generate_theoretical_data(theoretical, data_size, 0);
        if (verify_data(recv_data, theoretical, data_size)) {
            printf("Rank 1: Received data verified!\n");
        } else {
            printf("Rank 1: Verification failed!\n");
        }

        printf("Rank 1: Sending confirmation to rank 0...\n");
        MPI_Send(send_data, data_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        printf("Rank 1: Communication complete!\n");
    }

    free(send_data);
    free(recv_data);
    free(theoretical);

    MPI_Finalize();
    return 0;
}