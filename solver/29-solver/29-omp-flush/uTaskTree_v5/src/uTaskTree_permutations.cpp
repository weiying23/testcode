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

#include "uTaskTree_permutations.h"
#include <iostream>
#include <assert.h>
#include <algorithm>
#include "uTaskTree.h"

using namespace std;

void uTaskTree::uTaskTree_permute_int2d(int *tab, int *perm, int nbItem, int dimItem, int offset)
{
    bool *vis = new bool [nbItem] ();
    int  *tmpSrc    = new int  [dimItem];
    int  *tmpDst    = new int  [dimItem];

    for (int i = 0; i < nbItem; i++) {
        if (vis[i] == 1) continue;

        int init = i, src = i, dst;
        for (int j = 0; j < dimItem; j++) {
            tmpSrc[j] = tab[(i+offset)*dimItem+j];
        }
        do {
            dst = perm[src];
            for (int j = 0; j < dimItem; j++) {
                tmpDst[j] = tab[(dst+offset)*dimItem+j];
                tab[(dst+offset)*dimItem+j] = tmpSrc[j];
                tmpSrc[j] = tmpDst[j];
            }
            src = dst;
            vis[src] = 1;
        }
        while (src != init);
    }
    delete[] tmpDst, delete[] tmpSrc, delete[] vis;
}

void uTaskTree::merge_permutations (int *perm, int *localItemPerm, int globalNbItem, int localNbItem,
						 int firstItem, int lastItem)
{
    int ctr = 0;
    for (int i = 0; i < globalNbItem; i++) {
        int dst = perm[i];
        if (dst >= firstItem && dst <= lastItem) {
            perm[i] = localItemPerm[dst-firstItem] + firstItem;
            ctr++;
        }
        if (ctr == localNbItem)	break;
    }
}

// Create permutation array from partition array
void uTaskTree_create_permutation (int *perm, int *part, int size)
{
    pair<int,int> *a;
    a = new pair<int,int>[size];
    
    for (int i = 0; i < size; i++)
        a[i] = make_pair(part[i], i);
    
    sort(a, a + size);
    
    for (int i = 0; i < size; i++)
        perm[a[i].second] = i;
    
    delete[] a;
}

#ifdef DC_VEC
// Create coloring permutation array with full vectorial colors stored first &
// return the index of the last element in a full vectorial color
int create_coloring_permutation (int *perm, int *part, int *card, int size,
                                 int nbColors)
{
    int ptr = 0, lastFullColor;

    // Full colors
    for (int i = 0; i < nbColors; i++) {
        if (card[i] == VEC_SIZE) {
            for (int j = 0; j < size; j++) {
                if (part[j] == i) {
                    perm[j] = ptr;
                    ptr++;
                }
            }
        }
    }
    lastFullColor = ptr - 1;

    // Other colors
    for (int i = 0; i < nbColors; i++) {
        if (card[i] < VEC_SIZE) {
            for (int j = 0; j < size; j++) {
                if (part[j] == i) {
                    perm[j] = ptr;
                    ptr++;
                }
            }
        }
    }
    return lastFullColor;
}
#endif
