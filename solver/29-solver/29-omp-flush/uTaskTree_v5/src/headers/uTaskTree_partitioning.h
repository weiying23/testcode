#ifndef uTaskTree_partitioning_h
#define uTaskTree_partitioning_h

#include "uTaskTree.h"

void create_nodal_graph (int *graphIndex, int **graphValue, int *i2n, int localNbItem, int dimItem, int localNbNode);

int create_local_i2n (int *local_i2n, int *i2n, int firstItem, int lastItem, int dimItem);


#endif
