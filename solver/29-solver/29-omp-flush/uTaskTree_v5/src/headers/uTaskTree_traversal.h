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

#ifndef TREE_TRAVERSAL_H
#define TREE_TRAVERSAL_H

#include "uTaskTree.h"

void uTaskTree_traversal_forward (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *treePtr, int nbParts);

void uTaskTree_traversal_backward (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *tree, int nbParts);

void uTaskTree_traversal_noDependence (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *treePtr, int nbParts);

#endif
