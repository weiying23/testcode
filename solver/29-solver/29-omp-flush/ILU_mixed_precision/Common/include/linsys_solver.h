#ifndef LINSYS_SOLVER_H
#define LINSYS_SOLVER_H

#include "linsys_matrix.h"
#include "linsys_vector.h"
#include "grid_polyhedra.h"
namespace mflow
{

class LinSysSolver
{
public:
    enum TYPE
    {
        LUSGS, GMRES_PETSC
    };

private:
    IntType method;      /// the method of solving the linear system
    LinSysMatrix * matA; /// representing the matrix that defines the linear system
    PolyGrid * grid;     /// The matrix form depends on the elemenets' collection relation of grid topology 
#ifdef USING_PETSC
    KSP ksp;             /// solver of petsc
#endif
public:

    /// \param[in] vecb - Linear system residual
    /// \param[in,out] vecx - Linear system solution
    void Solve(const LinSysVector & vecb, LinSysVector & vecx);

    void SetGrid(PolyGrid * grid);

    PolyGrid * GetGrid() const ;

    void SetMethod(const LinSysSolver::TYPE method);

    IntType GetSize() const { return matA->getSize();}

    IntType GetNVar() const { return matA->GetNVar();}

#ifdef USING_PETSC
    void SetTolerances(RealFlow tol, IntType maxits);

    void SetRestart(IntType kspan);

    PetscErrorCode Solve(Vec & vecb, Vec & vecx);

    void SetPC(PetscErrorCode (*userPC)(PC,Vec,Vec));

    void SetPCContext(void *ctx);

    void SetPC(IntType pctype);

    KSP GetKSP() {return ksp;}
#endif
    ///constructor
    LinSysSolver(LinSysMatrix * matA);

    ~LinSysSolver();
private:
    /// \param[in] vecb - Linear system residual
    /// \param[in,out] vecx - Linear system solution
    void SolveLUSGS(const LinSysVector & vecb, LinSysVector & vecx);
};

inline void LinSysSolver::SetMethod(const LinSysSolver::TYPE method)
{
    this->method = method;
}

inline void LinSysSolver::SetGrid(PolyGrid * grid)
{
    this->grid = grid;
}

inline PolyGrid * LinSysSolver::GetGrid() const
{
    return grid;
}
}
#endif