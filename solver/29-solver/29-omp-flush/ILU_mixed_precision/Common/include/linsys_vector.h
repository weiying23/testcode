#ifndef LINSYS_VECTOR_H
#define LINSYS_VECTOR_H
#include "number_type.h"
#include <string.h>

#ifdef USING_PETSC
#include "petscksp.h"
#endif
namespace mflow
{

class LinSysVectorBase
{

};

class LinSysVector : public LinSysVectorBase
{
private:
    RealFlow * data; /// storage for the data of system vector
    RealFlow * ghostdata; /// storage for the ghost data
    IntType size;    /// length  of the data
    IntType ghostsize; /// length of the ghost data
    IntType nVar;    /// number of vars solved in the equation
    IntType nElem;   /// total number of cells or elements in this processor
    IntType nGhostElem;   /// number of ghost cells 

public:
    LinSysVector(IntType nElem, IntType nVar);

    /// construtor considering ghost element
    LinSysVector(IntType nElem, IntType nGhostElem, IntType nVar);

    LinSysVector(IntType size);

    ~LinSysVector();

public:

    IntType GetNVar() const { return nVar;}

    IntType GetNElem() const {return nElem;}

    /// \brief Access operator with assignment permitted.
    /// \param[in] i - Local index to access.
    inline RealFlow & operator[](IntType i) { return data[i]; }
    inline const RealFlow& operator[](IntType i) const { return data[i]; }

    inline RealFlow * getData() { return data;}

    inline RealFlow * getGhosts() const { return ghostdata;}

    /// \brief Access operator with assignment permitted block version.
    /// \param[in] iElem - Index of element.
    /// \param[in] iVar - Index of variable.
    inline RealFlow & operator()(IntType iElem, IntType iVar) { return data[iElem * nVar + iVar]; }
    inline const RealFlow & operator()(IntType iElem, IntType iVar) const { return data[iElem * nVar + iVar]; }

    inline void CopyDataDeep(const RealFlow * data) 
    {
        memcpy(this->data, data, size*sizeof(RealFlow));
    }
};

}
#endif