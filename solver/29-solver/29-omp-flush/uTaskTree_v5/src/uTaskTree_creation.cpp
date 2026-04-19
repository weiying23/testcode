/*  Copyright 2014 - UVSQ
    Authors list: LoÃ¯c ThÃ©bault, Eric Petit

    This file is part of the DC-lib.

    DC-lib is free software: you can redistribute it and/or modify it under the
    terms of the GNU Lesser General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later version.

    DC-lib is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along with
    the DC-lib. If not, see <http://www.gnu.org/licenses/>. */

#ifdef CILK
    #include <cilk/cilk.h>
#endif
#include <pthread.h>
#include <limits.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <unordered_map>
#include <vector>

#include "uTaskTree.h"
#include "uTaskTree_permutations.h"
#include "uTaskTree_partitioning.h"
#include "uTaskTree_creation.h"
#include "omp.h"

#ifdef TREE_CREATION

// Initialize the content of D&C tree nodes
void uTaskTree_node_init (uTaskTreeNode *treePtr, int firstCell, int lastCell, int firstFace, int lastFace, int firstNode, int lastNode, 
                            int nbIsoItem, int *nbPartItem, bool isIso, bool isLeaf, int nbParts)
{
    treePtr->firstCell    = firstCell;
    treePtr->lastCell     = lastCell;
    treePtr->firstFace    = firstFace;
    treePtr->lastFace     = lastFace;
    treePtr->firstNode    = firstNode;
    treePtr->lastNode     = lastNode;
    treePtr->isIso        = isIso;
    treePtr->iso          = nullptr;
	treePtr->son		  = nullptr;
	treePtr->isLeaf		  = isLeaf;
    treePtr->nbParts      = nbParts;

    // if (isLeaf)
        // cout << lastFace-firstFace+1 << ' ' << firstFace << ' ' << lastFace << ' ' << isIso << endl << flush;

    if (isLeaf == false) {
		
		treePtr->son = new uTaskTreeNode*[nbParts];
		// cout<<"1"<<endl<<flush;
		for (int i = 0; i < nbParts; i++)
			if (nbPartItem[i] > 0)
                treePtr->son[i] = new uTaskTreeNode();
			else 
				treePtr->son[i] = nullptr;

		// cout<<"2"<<endl<<flush;
        if (nbIsoItem > 0) 
            treePtr->iso = new uTaskTreeNode();
    }
}

void uTaskTree_create_itemPart (int *itemPart, int *nodePart, int *i2n, int localNbItem, int dimItem, int *nbPartItem, int *nbIsoItem)
{
	for (int i = 0; i < localNbItem; i++) {
		int node, colorA = -1, colorB = -1;
		for (int j = 0; j < dimItem; j++) {
			node = i2n[i*dimItem+j];
			if (colorA == -1) 
                colorA = nodePart[node];
            else
                if (colorA != nodePart[node]) 
                    colorB = nodePart[node];
		}
		if (colorB == -1) {
			nbPartItem[colorA]++;
			itemPart[i] = colorA;
		}
		else {
			(*nbIsoItem) ++;
			itemPart[i] = 1e9;
		}
	}
}

void uTaskTree_create_itemPart_accordingly(int *itemPerm, int *ItemPart, int *itemPart, int *I2i, int dimItem, 
                                int firstItem, int lastItem, int firstitem, int lastitem, int *nbPartitem, 
                                int *nbIsoitem)
{
    int localNbItem = lastItem - firstItem + 1;
    int localNbitem = lastitem - firstitem + 1;

    for(int i = 0; i < localNbitem; i++)
        itemPart[i] = 1e9;

    for(int i = 0; i < localNbItem; i++)
    {
        if (ItemPart[i] == 1e9) continue;
        for (int j = 0; j < dimItem; j++)
        {
            int item = itemPerm[I2i[(i+firstItem)*dimItem+j]];
            if (item < firstitem || item > lastitem) continue;
            item -= firstitem;
            itemPart[item] = ItemPart[i];
        }
    }

    for(int i = 0; i < localNbItem; i++)
    {
        if (ItemPart[i] != 1e9) continue;
        for (int j = 0; j < dimItem; j++)
        {
            int item = itemPerm[I2i[(i+firstItem)*dimItem+j]];
            if (item < firstitem || item > lastitem) continue;
            item -= firstitem;
            itemPart[item] = 1e9;
        }
    }
    
    for(int i = 0; i < localNbitem; i++)
        if (itemPart[i] == 1e9)
            (*nbIsoitem) ++;
        else
            nbPartitem[itemPart[i]] ++;
}

void uTaskTree::uTaskTree_create_boundary  (uTaskTreeNode *tree, int *f2n, int *f2c, int globalNbFace, int dimFace, 
                                                int firstFace, int lastFace, int globalNbNode,
                                                int firstNode, int lastNode, bool isIso)
{
    int localNbFace = lastFace - firstFace + 1;
    int localNbNode = lastNode - firstNode + 1;
  
    //cout << firstFace << ' ' << lastFace << ' ' << firstNode << ' ' << lastNode << endl << flush;

	if (localNbFace < PARTSIZE) {
        uTaskTree_node_init (tree, 0, -1, firstFace, lastFace, firstNode, lastNode, 0, 0, isIso, true, 0);
        return;
    }

    // cout << "tree_creation 0 " << endl << flush;

    int *nodePart = new int[localNbFace*dimFace];
    int *local_f2n = new int[localNbFace*dimFace];
    int parts = uTaskTree_partitioning (f2n, local_f2n, dimFace, firstFace, lastFace, nodePart);

    // cout << "tree_creation 1 " << endl << flush;

    int nbIsoFace = 0;
    int *nbPartFace = new int [parts]();

    int *facePart = new int [localNbFace];
    uTaskTree_create_itemPart (facePart, nodePart, local_f2n, localNbFace, dimFace, nbPartFace, &nbIsoFace);
    delete[] local_f2n;
    delete[] nodePart;

    //Correct some faces that share the same cell
    //Remain one of them unmoved, others become isolator.
    unordered_map<int,int> mp;
    for(int i = 0; i < localNbFace; i++)
    {
        if (facePart[i] == 1e9)
            continue;
        int face = i + firstFace;
        if (!mp.count(f2c[face+face])||mp[f2c[face+face]]==facePart[i])
            mp[f2c[face+face]]=facePart[i];
        else
        {
            nbPartFace[facePart[i]]--;
            facePart[i] = 1e9;
            nbIsoFace++;
        }
    }
    mp.clear();
    
    // cout << "tree_creation 2 " << endl << flush;

    int nbIsoNode = 0;
    int *nbPartNode = new int [parts]();
    
    if (localNbNode > 0)
    {
        nodePart = new int[localNbNode];
        uTaskTree_create_itemPart_accordingly (nodePerm, facePart, nodePart, f2n, dimFace, firstFace, lastFace, firstNode, lastNode, nbPartNode, &nbIsoNode);
        
        int *localNodePerm = new int [localNbNode];
        uTaskTree_create_permutation (localNodePerm, nodePart, localNbNode);
        delete[] nodePart;

        merge_permutations (nodePerm, localNodePerm, globalNbNode, localNbNode, firstNode, lastNode);
        delete[] localNodePerm;
    }

    int *localFacePerm = new int [localNbFace];
    uTaskTree_create_permutation (localFacePerm, facePart, localNbFace);
    delete[] facePart;

    // cout << "tree_creation 3 " << endl << flush;

    merge_permutations (facePerm, localFacePerm, globalNbFace, localNbFace, firstFace, lastFace);
   
    // cout << "tree_creation 4 " << endl << flush;

    uTaskTree_permute_int2d (f2n, localFacePerm, localNbFace, dimFace, firstFace);
    uTaskTree_permute_int2d (f2c, localFacePerm, localNbFace, 2, firstFace);
    delete[] localFacePerm;

    // cout << "tree_creation 5 " << parts << endl << flush;

    int nbIsoItem = max(nbIsoFace, nbIsoNode);
    int *nbPartItem = new int[parts]();
    for (int i = 0; i < parts; i++)
        nbPartItem[i] = max(nbPartFace[i], nbPartNode[i]);

    // cout << "tree_creation 5.5 " << endl << flush;
    uTaskTree_node_init (tree, 0, -1, firstFace, lastFace, firstNode, lastNode, nbIsoItem, nbPartItem, isIso, false, parts);

    // cout << "tree_creation 6 " << endl << flush;

    tree->isLeaf = true;
    int stFace[parts], edFace[parts], st2 = firstFace, ed2;
    int stNode[parts], edNode[parts], st3 = firstNode, ed3;
    for(int i = 0; i < parts; i++, st2 = ed2 + 1, st3 = ed3 + 1)
    {
        stFace[i] = st2;
        ed2 = edFace[i] = st2 + nbPartFace[i]-1;
        stNode[i] = st3;
        ed3 = edNode[i] = st3 + nbPartNode[i]-1;
    }

#ifndef FORKJOIN
#ifdef OMP
    #pragma omp taskloop default(shared)
    for (int i = 0; i < parts; i++) {
#elif CILK
    cilk_for (int i = 0; i < parts; i++) {
#endif
#else
    #pragma omp parallel for
    for (int i = 0; i < parts; i++) {
#endif
        if (nbPartItem[i] > 0)
        {
            tree->isLeaf = false;
#ifndef FORKJOIN
            uTaskTree_create_boundary (tree->son[i], f2n, f2c, globalNbFace, dimFace, stFace[i], edFace[i], globalNbNode, stNode[i], edNode[i], isIso);
#else
            uTaskTree_node_init (tree->son[i], 0, -1, stFace[i], edFace[i], stNode[i], edNode[i], 0, 0, isIso, true, parts);
#endif
        }
    }

#ifndef FORKJOIN
    // Synchronization
#ifdef OMP
        #pragma omp taskwait
#elif CILK
        cilk_sync;
#endif        
#endif
    // cout << "tree_creation 6.5 " << endl << flush;

    delete[] nbPartFace;
    delete[] nbPartNode;
    delete[] nbPartItem;
    
    // cout << "tree_creation 7 " << endl << flush;

    if (nbIsoItem > 0 && !tree->isLeaf) 
		uTaskTree_create_boundary (tree->iso, f2n, f2c, globalNbFace, dimFace, lastFace-nbIsoFace+1, lastFace, 
                                        globalNbNode, lastNode-nbIsoNode+1, lastNode, true);

    // cout << "tree_creation 8 " << endl << flush;

}           

// Create the D&C tree and the element permutation, and compute the intervals of nodes
// and elements at each node of the tree-
void uTaskTree::uTaskTree_create_normal  (uTaskTreeNode *tree, int *c2c, int *c2f, int *c2n, int globalNbCell, 
                                        int dimCell1, int dimCell2, int dimCell3, int firstCell, int lastCell, int globalNbFace, 
                                        int firstFace, int lastFace, int globalNbNode, int firstNode, int lastNode, 
                                        bool isIso, int level)
{
    int localNbCell = lastCell - firstCell + 1;
    int localNbFace = lastFace - firstFace + 1;
    int localNbNode = lastNode - firstNode + 1;
    //cout << firstCell << ' ' << lastCell << ' ' << firstFace << ' ' 
    //        << lastFace << ' ' << firstNode << ' ' << lastNode << endl << flush;

	if (localNbCell < PARTSIZE) {
        uTaskTree_node_init (tree, firstCell, lastCell, firstFace, lastFace, firstNode, lastNode, 0, 0, isIso, true, 0);
        return;
    }
    // cout << "tree_creation 0 " << endl << flush;

    int *tmpCellPart = new int[localNbCell*dimCell3];
    int *local_c2n = new int[localNbCell*dimCell3];
    int parts = uTaskTree_partitioning (c2n, local_c2n, dimCell3, firstCell, lastCell, tmpCellPart);

    // cout << "tree_creation 0.5 " << endl << flush;
    int nbIsoCell = 0;
    int *nbPartCell = new int [parts]();

    int *cellPart = new int [localNbCell];
    uTaskTree_create_itemPart (cellPart, tmpCellPart, local_c2n, localNbCell, dimCell3, nbPartCell, &nbIsoCell);
    delete[] local_c2n;
    delete[] tmpCellPart;

    unordered_map<int,int> mp;
    for(int i=firstCell;i<=lastCell;i++)
    {
        if (cellPart[i-firstCell]==1e9) continue;
        for(int j=0;j<dimCell1;j++)
            if (mp.count(c2c[i*dimCell1+j])&&mp[c2c[i*dimCell1+j]]!=cellPart[i-firstCell])
            {
                nbPartCell[cellPart[i-firstCell]]--;
                nbIsoCell++;
                cellPart[i-firstCell]=1e9;
                break;
            }
        if (cellPart[i-firstCell]==1e9) continue;
        for(int j=0;j<dimCell1;j++)
            mp[c2c[i*dimCell1+j]]=cellPart[i-firstCell];
    }

    //Correct some cells that share the same node
    //Remain one of them unmoved, others become isolator.
    int nbIsoFace = 0;
    int *nbPartFace = new int [parts]();
    if (localNbFace > 0)
    {
        int *facePart = new int[localNbFace];
        uTaskTree_create_itemPart_accordingly (facePerm, cellPart, facePart, c2f, dimCell2, firstCell, lastCell, firstFace, lastFace, nbPartFace, &nbIsoFace);
        
        int *localFacePerm = new int [localNbFace];
        uTaskTree_create_permutation (localFacePerm, facePart, localNbFace);
        delete[] facePart;

        merge_permutations (facePerm, localFacePerm, globalNbFace, localNbFace, firstFace, lastFace);
        delete[] localFacePerm;
    }

    int nbIsoNode = 0;
    int *nbPartNode = new int [parts]();
    if (localNbNode > 0)
    {
        int *nodePart = new int[localNbNode];
        uTaskTree_create_itemPart_accordingly (nodePerm, cellPart, nodePart, c2n, dimCell3, firstCell, lastCell, firstNode, lastNode, nbPartNode, &nbIsoNode);
        
        int *localNodePerm = new int [localNbNode];
        uTaskTree_create_permutation (localNodePerm, nodePart, localNbNode);
        delete[] nodePart;

        merge_permutations (nodePerm, localNodePerm, globalNbNode, localNbNode, firstNode, lastNode);
        delete[] localNodePerm;       
    }

    // cout << "tree_creation 2 " << endl << flush;

    // Create local element permutation
    int *localCellPerm = new int [localNbCell];
    uTaskTree_create_permutation (localCellPerm, cellPart, localNbCell);
    delete[] cellPart;

    // cout << "tree_creation 3 " << endl << flush;

    merge_permutations (cellPerm, localCellPerm, globalNbCell, localNbCell, firstCell, lastCell);
   
    // cout << "tree_creation 4 " << endl << flush;

    uTaskTree_permute_int2d (c2c, localCellPerm, localNbCell, dimCell1, firstCell);
    uTaskTree_permute_int2d (c2f, localCellPerm, localNbCell, dimCell2, firstCell);
    uTaskTree_permute_int2d (c2n, localCellPerm, localNbCell, dimCell3, firstCell);
    delete[] localCellPerm;

    // cout << "tree_creation 5 " << parts << endl << flush;

    int nbIsoItem = max(nbIsoCell, max(nbIsoFace, nbIsoNode));
    int *nbPartItem = new int[parts]();
    for (int i = 0; i < parts; i++)
        nbPartItem[i] = max(nbPartCell[i], max(nbPartFace[i], nbPartNode[i]));

    // Initialize current node    
    uTaskTree_node_init (tree, firstCell, lastCell, firstFace, lastFace, firstNode, lastNode, nbIsoItem, nbPartItem, isIso, false, parts);

    // cout << "tree_creation 6 " << nbIsoCell << endl << flush;

    tree->isLeaf = true;
    int stCell[parts], edCell[parts], st1 = firstCell, ed1;
    int stFace[parts], edFace[parts], st2 = firstFace, ed2;
    int stNode[parts], edNode[parts], st3 = firstNode, ed3;
    for(int i = 0; i < parts; i++, st1 = ed1+1, st2 = ed2 + 1, st3 = ed3 + 1)
    {
        stCell[i] = st1;
        ed1 = edCell[i] = st1 + nbPartCell[i]-1;
        stFace[i] = st2;
        ed2 = edFace[i] = st2 + nbPartFace[i]-1;
        stNode[i] = st3;
        ed3 = edNode[i] = st3 + nbPartNode[i]-1;
    }

#ifndef FORKJOIN
#ifdef OMP
    #pragma omp taskloop default(shared)
    for (int i = 0; i < parts; i++) {
#elif CILK
    cilk_for (int i = 0; i < parts; i++) {
#endif
#else
    #pragma omp parallel for
    for (int i = 0; i < parts; i++) {
#endif
        if (nbPartItem[i] > 0)
        {
            tree->isLeaf = false;
#ifndef FORKJOIN
            uTaskTree_create_normal (tree->son[i], c2c, c2f, c2n, globalNbCell, dimCell1, dimCell2, dimCell3, stCell[i], edCell[i],
                                globalNbFace, stFace[i], edFace[i], globalNbNode, stNode[i], edNode[i], isIso, level+1);
#else
            uTaskTree_node_init (tree->son[i], stCell[i], edCell[i], stFace[i], edFace[i], stNode[i], edNode[i], 0, 0, isIso, true, parts);
#endif
        }
    }
 
#ifndef FORKJOIN
    // Synchronization
    #ifdef OMP
        #pragma omp taskwait
    #elif CILK
        cilk_sync;
    #endif        
#endif

    // cout << "tree_creation 6.5 " << endl << flush;

    delete[] nbPartCell;
    delete[] nbPartFace;
    delete[] nbPartNode;
    delete[] nbPartItem;
    
    // cout << "tree_creation 7 " << endl << flush;

    // D&C partitioning of Isoarator elements
    if (nbIsoItem > 0 && !tree->isLeaf)
		uTaskTree_create_normal (tree->iso, c2c, c2f, c2n, globalNbCell, dimCell1, dimCell2, dimCell3, lastCell-nbIsoCell+1, lastCell, 
                                                                        globalNbFace, lastFace-nbIsoFace+1, lastFace, 
                                                                        globalNbNode, lastNode-nbIsoNode+1, lastNode, 
                                                                        true, level+1);
    else
    {
        tree->nbParts = 0;
        // cout << tree->firstCell << ' ' << tree->lastCell << endl << flush;
    }

    // cout << "tree_creation 8 " << endl << flush;

}

int CountIso(uTaskTreeNode *tree)
{
    if (tree->isIso) return tree->lastCell - tree->firstCell + 1;
    int res = 0;
    for(int i = 0; i < tree->nbParts; i++)
    {
        if (tree->son[i] != nullptr)
            res += CountIso(tree->son[i]);
    }
    if (tree->iso != nullptr)
        res += tree->iso->lastCell - tree->iso->firstCell + 1;
    return res;
}

int CountLeaf(uTaskTreeNode *tree)
{
    if (tree->isLeaf) return 1;
    int res = 0;
    for(int i = 0; i < tree->nbParts; i++)
    {
        if (tree->son[i] != nullptr)
            res += CountLeaf(tree->son[i]);
    }
    if (tree->iso != nullptr)
        res += CountLeaf(tree->iso);
    return res;
}

int CountIsoLeaf(uTaskTreeNode *tree)
{
    if (tree->isLeaf) return 0;
    int res = 0;
    for(int i = 0; i < tree->nbParts; i++)
    {
        if (tree->son[i] != nullptr)
            res += CountIsoLeaf(tree->son[i]);
    }
    if (tree->iso != nullptr)
        res += CountLeaf(tree->iso);
    return res;
}

void CountLevel(uTaskTreeNode *tree)
{
    if (tree->isLeaf) 
    {
        cout << tree->firstCell << " " << tree->lastCell << endl;
        return;
    }
    cout << tree->firstCell << " " << tree->son[tree->nbParts-1]->lastCell << endl;
    if (tree->iso != nullptr)
        CountLevel(tree->iso);
}

// Create the D&C tree and the permutations
// qleonrdo: dimElem can be different
//void DC_create_tree (int *elemToNode, int nbElem, int dimElem, int nbNodes)
void uTaskTree::uTaskTree_creation (int *c2c, int *c2f, int *c2n, int *f2n, int *f2c, int globalNbCell, int dimCell1, int dimCell2, int dimCell3, 
                                    int globalNbFace, int dimFace, int globalNbNode, int nPFace, int nBFace)
{
    double start = omp_get_wtime();
#ifdef OMP

    #pragma omp parallel for
    for (int i = 0; i < globalNbCell; i++) 
        cellPerm[i] = i;

    #pragma omp parallel for
    for (int i = 0; i < globalNbFace; i++) 
        facePerm[i] = i;

    #pragma omp parallel for
    for (int i = 0; i < globalNbNode; i++) 
        nodePerm[i] = i;

#elif CILK

    cilk_for (int i = 0; i < globalNbCell; i++) 
        cellPerm[i] = i;
        
    cilk_for (int i = 0; i < globalNbFace; i++) 
        facePerm[i] = i;
        
    cilk_for (int i = 0; i < globalNbNode; i++) 
        nodePerm[i] = i;

#endif
    // cout << globalNbCell << ' ' << globalNbFace << ' ' << globalNbNode << endl;

    // int nbIsoFace = globalNbFace - nPFace;
    // int *nbPartFace = new int [1]();
    // int *facePart = new int [globalNbFace]();
    
    // nbPartFace[0] = nPFace;

    // uTaskTree_node_init (treeRoot, 0, globalNbCell-1, 0, globalNbFace-1, 0, globalNbNode-1, nbIsoFace, nbPartFace, false, false, 1);

    // if (nPFace > 0)
    //     uTaskTree_create_boundary (treeRoot->son[0], f2n, f2c, globalNbFace, dimFace, 0, nPFace-1, globalNbNode, 0, -1, false);

    // nbIsoFace = globalNbFace - nBFace;
    // facePart = new int [globalNbFace-nPFace]();

    // nbPartFace[0] = nBFace - nPFace;

    // uTaskTree_node_init (treeRoot->iso, 0, globalNbCell-1, nPFace, globalNbFace-1, 0, globalNbNode-1, nbIsoFace, nbPartFace, false, false, 1);

    // if (nBFace - nPFace > 0)
    //     uTaskTree_create_boundary (treeRoot->iso->son[0], f2n, f2c, globalNbFace, dimFace, nPFace, nBFace-1, globalNbNode, 0, -1, false);

#ifndef FORKJOIN
#ifdef OMP
        #pragma omp parallel
        #pragma omp single nowait
#endif
#endif
    uTaskTree_create_normal (treeRoot, c2c, c2f, c2n, globalNbCell, dimCell1, dimCell2, dimCell3, 0, globalNbCell-1, globalNbFace, nBFace, globalNbFace-1,
                                                            globalNbNode, 0, globalNbNode-1, false, 0);

    // cout << "build finish!!!!!!!!!!" << endl << flush;

// #ifndef FORKJOIN
//     cout << CountIso(treeRoot->iso->iso) << " " << globalNbCell << endl << flush;
//     cout << CountIsoLeaf(treeRoot->iso->iso) << " " << CountLeaf(treeRoot->iso->iso) << endl << flush;
// #else
//     CountLevel(treeRoot->iso->iso);
// #endif
    int *t = new int[globalNbCell]();
    for (int i = 0; i < globalNbCell; i++) 
        t[cellPerm[i]]++;
    for (int i = 0; i < globalNbCell; i++) 
        assert(t[i]==1);
    delete[] t;

    t = new int[globalNbFace]();
    for (int i = 0; i < globalNbFace; i++) 
        t[facePerm[i]]++;
    for (int i = 0; i < globalNbFace; i++) 
        assert(t[i]==1);
    delete[] t;

    t = new int[globalNbNode]();
    for (int i = 0; i < globalNbNode; i++) 
        t[nodePerm[i]]++;
    for (int i = 0; i < globalNbNode; i++) 
        assert(t[i]==1);
    delete[] t;

#ifdef OMP
        #pragma omp parallel for
        for (int i = 0; i < globalNbCell; i++) 
            cellRev[cellPerm[i]] = i;
            
        #pragma omp parallel for
        for (int i = 0; i < globalNbFace; i++) 
            faceRev[facePerm[i]] = i;
            
        #pragma omp parallel for
        for (int i = 0; i < globalNbNode; i++) 
            nodeRev[nodePerm[i]] = i;

#elif CILK
        cilk_for (int i = 0; i < globalNbCell; i++) 
            cellRev[cellPerm[i]] = i;
            
        cilk_for (int i = 0; i < globalNbFace; i++) 
            faceRev[facePerm[i]] = i;
            
        cilk_for (int i = 0; i < globalNbNode; i++) 
            nodeRev[nodePerm[i]] = i;

#endif

    double end = omp_get_wtime();
    cout << "Build Tree Time: " << end-start << endl << flush;
}

int* uTaskTree::uTaskTree_get_cellPerm()
{
    return cellPerm;
}

int* uTaskTree::uTaskTree_get_cellRev()
{
    return cellRev;
}
int* uTaskTree::uTaskTree_get_facePerm()
{
    return facePerm;
}

int* uTaskTree::uTaskTree_get_faceRev()
{
    return faceRev;
}

int* uTaskTree::uTaskTree_get_nodePerm()
{
    return nodePerm;
}

int* uTaskTree::uTaskTree_get_nodeRev()
{
    return nodeRev;
}


#endif
