#include "linsys_vector.h"
#include "memory_util.h"
namespace mflow
{

LinSysVector::LinSysVector( IntType nElem, IntType nVar )
    : data(NULL), ghostdata(NULL)
{
    size = nElem*nVar;
    mfmem::snew_array_1D(data, size, dmrfl);
    memset(data, 0, size*sizeof(RealFlow));
}

LinSysVector::LinSysVector( IntType nElem, IntType nGhostElem, IntType nVar )
    : data(NULL), ghostdata(NULL)
{
    size = nElem * nVar;
    ghostsize = nGhostElem * nVar;
    mfmem::snew_array_1D(data, size, dmrfl);
    mfmem::snew_array_1D(ghostdata, ghostsize, dmrfl);
    memset(data, 0, size*sizeof(RealFlow));
    memset(ghostdata, 0 ,ghostsize*sizeof(RealFlow));
}

LinSysVector::LinSysVector( IntType size )
    : data(NULL), ghostdata(NULL)
{
    this->size = size;
    mfmem::snew_array_1D(this->data, size, dmrfl);
    memset(data, 0, size*sizeof(RealFlow));
}

LinSysVector::~LinSysVector()
{
    mfmem::sdel_array_1D(data);
    mfmem::sdel_array_1D(ghostdata);
}

}

