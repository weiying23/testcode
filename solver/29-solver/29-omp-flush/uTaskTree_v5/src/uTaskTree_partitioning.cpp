#ifdef TREE_CREATION

#include <cstring>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <omp.h>
#include <unordered_map>
#ifdef CILK
    #include <cilk/cilk.h>
#endif
#include <pthread.h>
#include <metis.h>

#include "uTaskTree.h"
#include "uTaskTree_permutations.h"
#include "uTaskTree_creation.h"
#include "uTaskTree_partitioning.h"

// Create a nodal graph from a item mesh for METIS
void create_nodal_graph (int *graphIndex, int **graphValue, int *i2n, int localNbItem, int dimItem, int localNbNode)
{
    unordered_map<int, bool> *n2n = new unordered_map<int, bool>[localNbNode]();

    for (int i = 0; i < localNbItem; i++)
    {
        for (int j = 0; j < dimItem; j++)
        {
            if (j && i2n[i*dimItem+j] == i2n[i*dimItem]) break;
            for (int k = 0; k < dimItem; k++)
            {
                if (k && i2n[i*dimItem+k] == i2n[i*dimItem]) break;
                n2n[i2n[i*dimItem+j]][i2n[i*dimItem+k]] = 1;
                n2n[i2n[i*dimItem+k]][i2n[i*dimItem+j]] = 1; 
            }
        }
    }

    graphIndex[0] = 0;
    for (int i = 0; i < localNbNode; i++) 
        graphIndex[i+1] = graphIndex[i] + n2n[i].size();
    (*graphValue) = new int[graphIndex[localNbNode]];
    for(int i = 0, j = 0; i < localNbNode; i++)
    {
        for (auto tmp:n2n[i])
        {
            (*graphValue)[j++] = tmp.first;
            // cout << tmp.first << " ";
        }
        // cout << endl << flush;
        n2n[i].clear();
    }
    delete[] n2n;
}

// Create local itemToNode array containing elements indexed contiguously from 0 to
// localNbItem and return the number of nodes
int create_local_i2n (int *local_i2n, int *i2n, int firstItem, int lastItem, int dimItem)
{
    int localNbNode = 0;
    unordered_map<int, int> newNode;

    for (int i = firstItem*dimItem, j = 0; i < (lastItem+1)*dimItem; i++, j++) {
        int oldNode = i2n[i];
        if (!newNode.count(oldNode)) {
            newNode[oldNode] = localNbNode++;
        }
        local_i2n[j] = newNode[oldNode];
    }
    newNode.clear();
    return localNbNode;
}

void PartGraphKway(int *graphIndex, int *graphValue, int *cellPart, int n, int k)
{
    int maxCell = n / k + 1;
    int *q = new int[maxCell];
    for(int i = 0; i < n; i++) cellPart[i] = -1;
    for(int i = 0, first = 0; i < k; i++, first++)
    {
        int nbCell = 0, now = 0, flag;
        do
        {
            flag = 0;
            while(first < n)
            {
                if (cellPart[first] == -1)
                {
                    q[nbCell++] = first;
                    cellPart[first] = i;
                    flag = 1;
                    break;
                }
                first++;
            }
            while(now < nbCell)
            {
                int cell = q[now++];
                if (nbCell >= maxCell) break;
                
                // 使用CSR格式获取邻居
                int start = graphIndex[cell];
                int end = graphIndex[cell + 1];
                
                for(int j = start; j < end; j++)
                {
                    int nextCell = graphValue[j];  // graphValue中存储的是邻居节点ID
                    if (cellPart[nextCell] == -1)
                    {
                        q[nbCell++] = nextCell;
                        cellPart[nextCell] = i;
                        if (nbCell >= maxCell) break;
                    }
                }
            }
        } while (nbCell < maxCell && flag == 1);
    }
    delete[] q;
}

// D&C partitioning of separators with more than MAX_ELEM_PER_PART elements
// D&C partitioning of separators with more than MAX_ELEM_PER_PART elements
int uTaskTree::uTaskTree_partitioning (int *i2n, int *local_i2n, int dimItem, int firstItem, int lastItem, int *nodePart)
{
    int localNbItem = lastItem - firstItem + 1;
    int localNbNode = create_local_i2n (local_i2n, i2n, firstItem, lastItem, dimItem);
    
    // cout << "partitioning 1 " << localNbNode << endl << flush;

    int constraint = 1, objVal;
    int *graphIndex = new int [localNbNode + 1](), *graphValue;
    
    create_nodal_graph (graphIndex, &graphValue, local_i2n, localNbItem, dimItem, localNbNode);
      
    // cout << "partitioning  " <<  graphIndex[localNbNode] << endl << flush;
    int parts = nbParts;

// while(1!=0);
    // Execution is correct without mutex although cilkscreen detects many race
    // conditions. Check if the problem is solved with future version of METIS (5.0)
#ifdef OMP
    #pragma omp critical
#endif
    // 调用修改后的PartGraphKway函数，使用CSR格式
    PartGraphKway(graphIndex, graphValue, nodePart, localNbNode, parts);

    // cout << "partitioning 3 " << endl << flush;

// while(1!=0);
    delete[] graphValue, delete[] graphIndex;

    return parts;
    
    // cout << "partitioning 4 " << endl << flush;
}

#endif
