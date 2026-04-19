#include "linsys_matrix.h"
#include "io_log.h"
#include "memory_util.h"
#include <assert.h>
namespace mflow
{

void LinSysMatrix::UpperProduct( const LinSysVector &vec, IntType row, RealFlow * prod )
{
    if(type == LinSysMatrix::MATRIX_FREE)
    {
        pUpperProduct(*this, vec, row, prod);
    }
    else
    {
        mflog::log<<"Need new code!!!"<<std::endl;
    }
}

void LinSysMatrix::LowerProduct( const LinSysVector &vec, IntType row, RealFlow * prod )
{
    if(type == LinSysMatrix::MATRIX_FREE)
    {
        pLowerProduct(*this, vec, row, prod);
    }
    else
    {
        mflog::log<<"Need new code!!!"<<std::endl;
    }
}

int LinSysMatrix::SetUpperProduct( void (*pUpperProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod) )
{
    this->pUpperProduct = pUpperProduct;
    return 0;
}

int LinSysMatrix::SetLowerProduct( void (*pLowerProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod) )
{
    this->pLowerProduct = pLowerProduct;
    return 0;
}

void LinSysMatrix::SetCtx( void * ctx )
{
    this->ctx = ctx;
#ifdef USING_PETSC
    if(type == LinSysMatrix::MATRIX_FREE_PETSC)
    {
        MatShellSetContext(petscMatrix, ctx);
    }
#endif
}


void LinSysMatrix::SetValue( IntType iPos, IntType jPos, RealFlow val )
{
#ifdef USING_PETSC
    MatSetValue(petscMatrix,iPos,jPos,val,INSERT_VALUES);
#endif
#ifdef USING_VIENNACL
    viennaclMatrix[iPos][jPos] = val;
#endif
}

void LinSysMatrix::SetBlockValue( IntType iBlock, IntType jBlock, RealFlow *val )
{
#ifdef USING_PETSC
    MatSetValuesBlocked(petscMatrix,1,&iBlock,1,&jBlock,val,INSERT_VALUES); 
#endif

#ifdef USING_VIENNACL
    //// different from petsc, vienncl's iBlock and jBlock is the local index
    IntType count = 0;
    for(IntType iVar = 0; iVar < nVar; iVar++)
    {
        IntType row = iBlock*nVar + iVar;
        for(IntType jVar = 0; jVar < nVar;jVar++)
        {
            IntType col = jBlock*nVar + jVar;
            viennaclMatrix[row][col] = val[count++];
        }
    }
#endif
}

LinSysMatrix::LinSysMatrix( IntType nElem, IntType nVar, void *ctx, LinSysMatrix::TYPE type )
    : nElem(nElem), nVar(nVar), ctx(ctx),type(type), Istart(0), Iend(0), Bstart(0), Bend(0)
{
    size = nElem*nVar;
#ifdef USING_PETSC
    if(type == LinSysMatrix::MATRIX_FREE_PETSC)
    {
        MatCreateShell(PETSC_COMM_WORLD, size, size, PETSC_DECIDE , PETSC_DECIDE, ctx, &petscMatrix);
    }
    else
    {
        MatCreate(PETSC_COMM_WORLD, &petscMatrix);
        MatSetSizes(petscMatrix, size,size,PETSC_DECIDE,PETSC_DECIDE);
        //MatSetType(petscMatrix, MATSEQAIJ);
        MatSetFromOptions(petscMatrix);
        //MatGetOwnershipRange(petscMatrix,&Istart,&Iend);
        //Bstart = Istart/nVar;
        //Bend = Iend/nVar;
    }
#endif

#ifdef USING_VIENNACL
    viennaclMatrix.resize(size);
#endif
}

LinSysMatrix::LinSysMatrix( IntType nElem, IntType nVar, IntType * d_nnz, IntType *o_nnz, LinSysMatrix::TYPE type )
    : nElem(nElem), nVar(nVar),type(type)
{
    size = nElem * nVar;
#ifdef USING_PETSC
    assert(type == LinSysMatrix::BLOCK_MATRIX_PETSC);

#ifdef FS_CUDA
    MatCreate(PETSC_COMM_WORLD, &petscMatrix);
    MatSetSizes(petscMatrix, size,size,PETSC_DECIDE,PETSC_DECIDE);
    MatSetType(petscMatrix, MATMPIAIJ);
    MatSetFromOptions(petscMatrix);
    MatXAIJSetPreallocation(petscMatrix, nVar,d_nnz,o_nnz, NULL, NULL);
    //MatCreateAIJCUSPARSE(PETSC_COMM_WORLD,size,size,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,d_nnz,PETSC_DECIDE,o_nnz,&petscMatrix);
    //MatSetBlockSize(petscMatrix, nVar);
#else
    MatCreateBAIJ(PETSC_COMM_WORLD,nVar,size,size,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,d_nnz,PETSC_DECIDE,o_nnz,&petscMatrix);
#endif

    MatSetUp(petscMatrix);
    MatGetOwnershipRange(petscMatrix,&Istart,&Iend);
    Bstart = Istart/nVar;
    Bend = Iend/nVar;
#endif
}

LinSysMatrix::~LinSysMatrix()
{
#ifdef USING_PETSC
    if(type == MATRIX_FREE_PETSC || type == BLOCK_MATRIX_PETSC)
    {
        MatDestroy(&petscMatrix);
    }
#endif
}


void LinSysMatrix::AssemblyMatPetsc()
{
#ifdef USING_PETSC
    MatAssemblyBegin(petscMatrix,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petscMatrix,MAT_FINAL_ASSEMBLY);
#endif
}

void LinSysMatrix::PreallocationCOO(IntType count, IntType *oor, IntType *ooc)
{
#ifdef USING_PETSC
#ifdef PETSC_COO
    IntType ierr = MatSetPreallocationCOO(petscMatrix, count, oor, ooc);
#endif
#endif
}

void  LinSysMatrix::SetValuesCOO(RealFlow *v)
{
#ifdef USING_PETSC
#ifdef PETSC_COO
    IntType ierr = MatSetValuesCOO(petscMatrix, v, INSERT_VALUES);
#endif
#endif
}

#ifdef USING_PETSC
int LinSysMatrix::SetMatOpMultPetsc( void (*userMatMult)(void) )
{
    return MatShellSetOperation(petscMatrix, MATOP_MULT, userMatMult);
}

#endif

}

