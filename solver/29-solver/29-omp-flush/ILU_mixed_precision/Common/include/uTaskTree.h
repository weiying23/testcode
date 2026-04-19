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

#ifndef DC_H
#define DC_H

#include <stdint.h>
#include <string>
#include <vector>

#define Forward 0
#define Backward 1
#define noDependence 2
#define CELL 3
#define FACE 4 

// D&C tree structure
struct uTaskTreeNode 
{
	int nbParts;
	bool isIso, isLeaf;
	int firstCell, lastCell, firstFace, lastFace, firstNode, lastNode, vecOffset;
    struct uTaskTreeNode **son, *iso;
	int *boundryCell, *boundryNode;
	int boundryNbCell, boundryNbNode;
};

// D&C arguments structure
struct uTaskTreeArgs 
{
    int firstCell, lastCell, firstFace, lastFace, firstNode, lastNode;
		int **boundryCell, **boundryNode;
	int *boundryNbCell, *boundryNbNode;
};

typedef struct 
{
    int *index, *value;
} index_t;

using namespace std;

class uTaskTree
{
	private:
		uTaskTreeNode *treeRoot;
		int *cellPerm, *cellRev;
		int *facePerm, *faceRev;
		int *nodePerm, *nodeRev;
		int nbParts, PARTSIZE;
	
		int uTaskTree_partitioning (int *i2n, int *local_i2n, int dimItem, int firstItem, int lastItem, int *nodePart);

		void uTaskTree_create_boundary  (uTaskTreeNode *tree, int *f2n, int *f2c, int globalNbFace, int dimFace, 
                                                int firstFace, int lastFace, int globalNbNode,
                                                int firstNode, int lastNode, bool isIso);

		void uTaskTree_create_normal  (uTaskTreeNode *tree, int *c2c, int *c2f, int *c2n, int globalNbCell, 
                                        int dimCell1, int dimCell2, int dimCell3, int firstCell, int lastCell, int globalNbFace, 
                                        int firstFace, int lastFace, int globalNbNode, int firstNode, int lastNode, 
                                        bool isIso, int level);
		
		// Permute "tab" 2D array of int using "perm"
		void uTaskTree_permute_int2d (int *tab, int *perm, int nbItem, int dimItem, int offset);
									  
		// Apply local element permutation to global element permutation
		void merge_permutations (int *perm, int *localPerm, int globalNbItem, int localNbItem, int firstItem, int lastItem);

		void uTaskTree_face_coloring (uTaskTreeNode treePtr, index_t &n2f, int *f2n, int globalNbFace, int dimFace);

		void uTaskTree_cell_coloring (uTaskTreeNode treePtr, index_t &n2c, int *c2n, int globalNbCell, int dimCell);
	public:
	
		void uTaskTree_creation (int *c2c, int *c2f, int *c2n, int *f2n, int *f2c, int globalNbCell, int dimCell1, int dimCell2, 
                                    int dimCell3, int globalNbFace, int dimFace, int globalNbNode, int nPFace, int nBFace);

		void task_traversal (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *), 
                                void (*userVecFctPtr)  (char **, uTaskTreeArgs *), 
                                char **userArgs, int traversal_type, int *f2c=NULL, int nBFace=0);

		int* uTaskTree_get_cellPerm();

		int* uTaskTree_get_cellRev();

		int* uTaskTree_get_facePerm();

		int* uTaskTree_get_faceRev();
		
		int* uTaskTree_get_nodePerm();

		int* uTaskTree_get_nodeRev();
							
		uTaskTree(int globalNbCell, int globalNbFace, int globalNbNode, int nbparts, int partSize) 
		{
			treeRoot = new uTaskTreeNode();
			
			cellPerm = new int [globalNbCell];
			cellRev = new int [globalNbCell];
			facePerm = new int[globalNbFace];
			faceRev = new int [globalNbFace];
			nodePerm = new int [globalNbNode];
			nodeRev = new int [globalNbNode];

			nbParts = nbparts;
			PARTSIZE = partSize;
		}

	
};

// Create permutation array from partition array
void uTaskTree_create_permutation (int *perm, int *part, int size);

#endif
