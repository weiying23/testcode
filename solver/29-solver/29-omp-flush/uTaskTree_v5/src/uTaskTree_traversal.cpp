#ifdef CILK
    #include <cilk/cilk.h>
#elif OMP
    #include <omp.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "uTaskTree_traversal.h"

using namespace std;

// Follow the uTaskTree forward method to execute the given function in parallel tasks
void uTaskTree_traversal_forward (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *treePtr, int *f2c, int nBFace)
{
//#pragma omp critical
//cout << treePtr->firstCell << ' ' << treePtr->lastCell << ' ' << treePtr->firstFace << ' ' << treePtr->lastFace << ' ' << treePtr->firstNode << ' ' << treePtr->lastNode << endl << flush;
    // If current node is a leaf, call the appropriate function
    if (treePtr->isLeaf) 
    {

        // Initialize the uTaskTree arguments
        uTaskTreeArgs treeArgs;
        
		treeArgs.firstCell  = treePtr->firstCell;
		treeArgs.lastCell   = treePtr->lastCell;
        treeArgs.firstFace  = treePtr->firstFace;
        treeArgs.lastFace   = treePtr->lastFace;
        treeArgs.firstNode  = treePtr->firstNode;
        treeArgs.lastNode   = treePtr->lastNode;

         

#ifdef VEC
		// Call user vectorized function if neccessary
        if (userVecFctPtr != NULL)
        {
            treeArgs.lastFace   = treePtr->vecOffset;

            userVecFctPtr (userArgs, &treeArgs);

            treeArgs.firstFace  = treePtr->vecOffset+1;
            treeArgs.lastFace   = treePtr->lastFace;
        }
#endif

		// Call user sequential function
		userSeqFctPtr (userArgs, &treeArgs);     
        
    }
    else 
    {
        // Forward method: first execute sons and then execute isolator
        int nbParts = treePtr->nbParts;
#ifndef FORKJOIN
#ifdef OMP
#pragma omp taskloop default(shared)
		for (int i = 0; i < nbParts; i++)
#elif CILK
        cilk_for(int i = 0; i < nbParts; i++)
#endif
#else
        #pragma omp parallel for 
		for (int i = 0; i < nbParts; i++)
#endif
		{
            if (treePtr->son[i] != nullptr)
                uTaskTree_traversal_forward (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->son[i], f2c, nBFace);
		}
#ifndef FORKJOIN
        // Synchronization
#ifdef OMP
        #pragma omp taskwait
#elif CILK
        cilk_sync;
#endif
#endif
        if (treePtr->iso != nullptr) {
            uTaskTree_traversal_forward (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->iso, f2c, nBFace);
        }
    }
}

// Follow the uTaskTree backward method to execute the given function in parallel tasks
void uTaskTree_traversal_backward (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *treePtr)
{
    // If current node is a leaf, call the appropriate function
    if (treePtr->isLeaf) 
    {

        // Initialize the uTaskTree arguments
        uTaskTreeArgs treeArgs;
        
		treeArgs.firstCell  = treePtr->firstCell;
		treeArgs.lastCell   = treePtr->lastCell;
        treeArgs.firstFace  = treePtr->firstFace;
        treeArgs.lastFace   = treePtr->lastFace;
        treeArgs.firstNode  = treePtr->firstNode;
        treeArgs.lastNode   = treePtr->lastNode;

#ifdef VEC
		// Call user vectorized function if neccessary
        if (userVecFctPtr != NULL)
        {
            treeArgs.lastFace   = treePtr->vecOffset;

            userVecFctPtr (userArgs, &treeArgs);

            treeArgs.firstFace  = treePtr->vecOffset+1;
            treeArgs.lastFace   = treePtr->lastFace;
        }
#endif

		// Call user sequential function
		userSeqFctPtr (userArgs, &treeArgs);     
        
    }
    else 
    {

        int nbParts = treePtr->nbParts;
        // Backward method: first execute isolator and then execute sons
        if (treePtr->iso != nullptr) {
            uTaskTree_traversal_backward (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->iso);
        }

#ifndef FORKJOIN
        // Synchronization
#ifdef OMP
        #pragma omp taskwait
#elif CILK
        cilk_sync;
#endif
#endif

#ifndef FORKJOIN    
#ifdef OMP
#pragma omp taskloop default(shared)
		for (int i = 0; i < nbParts; i++)
#elif CILK
        cilk_for(int i = 0; i < nbParts; i++)
#endif
#else
        #pragma omp parallel for default(shared)
		for (int i = 0; i < nbParts; i++)
#endif
		{
            if (treePtr->son[i] != nullptr)
                uTaskTree_traversal_backward (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->son[i]);
		}

    }
}

// Follow the uTaskTree forward method to execute the given function in parallel tasks
void uTaskTree_traversal_noDependence (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *),
                     void (*userVecFctPtr)  (char **, uTaskTreeArgs *),
                     char **userArgs, uTaskTreeNode *treePtr)
{
    // If current node is a leaf, call the appropriate function
    if (treePtr->isLeaf) 
    {

        // Initialize the uTaskTree arguments
        uTaskTreeArgs treeArgs;
        
		treeArgs.firstCell  = treePtr->firstCell;
		treeArgs.lastCell   = treePtr->lastCell;
        treeArgs.firstFace  = treePtr->firstFace;
        treeArgs.lastFace   = treePtr->lastFace;
        treeArgs.firstNode  = treePtr->firstNode;
        treeArgs.lastNode   = treePtr->lastNode;

#ifdef VEC
		// Call user vectorized function if neccessary
        if (userVecFctPtr != NULL)
        {
            treeArgs.lastFace   = treePtr->vecOffset;

            userVecFctPtr (userArgs, &treeArgs);

            treeArgs.firstFace  = treePtr->vecOffset+1;
            treeArgs.lastFace   = treePtr->lastFace;
        }
#endif

		// Call user sequential function
		userSeqFctPtr (userArgs, &treeArgs);     
        
    }
    else {
        
        int nbParts = treePtr->nbParts;
        // noDependence method: execute sons and isolator at the same time
#ifndef FORKJOIN
#ifdef OMP
#pragma omp taskloop default(shared)
		for (int i = 0; i < nbParts; i++)
#elif CILK
        cilk_for(int i = 0; i < nbParts; i++)
#endif
#else
        #pragma omp parallel for 
		for (int i = 0; i < nbParts; i++)
#endif
		{
            if (treePtr->son[i] != nullptr)
                uTaskTree_traversal_noDependence (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->son[i]);
		}
        
        if (treePtr->iso != nullptr) {
            uTaskTree_traversal_noDependence (userSeqFctPtr, userVecFctPtr, userArgs, treePtr->iso);
        }
    }
}

// Wrapper used to get the root of the D&C tree before calling the real tree traversal
void uTaskTree::task_traversal (void (*userSeqFctPtr)  (char **, uTaskTreeArgs *), 
                                void (*userVecFctPtr)  (char **, uTaskTreeArgs *), 
                                char **userArgs, int traversal_type, int *f2c, int nBFace)
{

#ifndef FORKJOIN
#ifdef OMP
        #pragma omp parallel
        #pragma omp single nowait
#endif
#endif
    if (traversal_type == Forward)
        uTaskTree_traversal_forward (userSeqFctPtr, userVecFctPtr, userArgs, treeRoot, f2c, nBFace);
    else 
        if (traversal_type == Backward)
            uTaskTree_traversal_backward (userSeqFctPtr, userVecFctPtr, userArgs, treeRoot);
        else
            uTaskTree_traversal_noDependence (userSeqFctPtr, userVecFctPtr, userArgs, treeRoot);         
}
