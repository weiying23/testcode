#include "linsys_solver.h"
#include "memory_util.h"
#include "linsys_preconditioner.h"
namespace mflow
{

void LinSysSolver::Solve( const LinSysVector & vecb, LinSysVector & vecx )
{
    switch (method)
    {
    case LUSGS:
        SolveLUSGS(vecb, vecx);
        break;
    default:
        break;
    }
}


LinSysSolver::LinSysSolver( LinSysMatrix * matA ):
    matA(matA)
{
#ifdef USING_PETSC
    if(matA->GetType() == LinSysMatrix::TYPE::MATRIX_FREE_PETSC || matA->GetType() == LinSysMatrix::TYPE::BLOCK_MATRIX_PETSC)
    {
        KSPCreate(PETSC_COMM_WORLD, &ksp);
        Mat A = matA->GetMatPetsc();
        KSPSetOperators(ksp, A, A);
    }
#endif
}

LinSysSolver::~LinSysSolver()
{
#ifdef USING_PETSC
    if(matA->GetType() == LinSysMatrix::TYPE::MATRIX_FREE_PETSC || matA->GetType() == LinSysMatrix::TYPE::BLOCK_MATRIX_PETSC)
    {
        KSPDestroy(&ksp);
    }
#endif
}

void LinSysSolver::SolveLUSGS( const LinSysVector & vecb, LinSysVector & vecx )
{
    IntType nElem = matA->GetNElem();
    IntType nVar  = matA->GetNVar();
    // foward sweep 
    
    RealFlow * flux_Jacobian = NULL;
    mfmem::snew_array_1D(flux_Jacobian, nVar, dmrfl);
    for(IntType ilu = 0; ilu < nElem; ilu++)
    {
        // Compute L.x*
        matA->LowerProduct(vecx,ilu,flux_Jacobian);
        RealFlow diag = matA->GetDiag(ilu);
        // Compute y = b - L.x*
        // then solve D.x* = y
        IntType idx = ilu * nVar;
        for(IntType iVar = 0; iVar < nVar; iVar++)
        {
            vecx[idx+iVar] = (vecb[idx+iVar] - flux_Jacobian[iVar])/diag;
        }
    }
    
#ifdef MPICH
    // Communicate dq
    RealFlow * vecx_local = vecx.getData();
    RealFlow * vecx_ghosts = vecx.getGhosts();
    grid->UpdateVectorGhostVar(vecx_local, vecx_ghosts, nVar);
    
#endif

    // backward sweep:
    for(IntType ilu=nElem-1; ilu>=0; ilu--)
    {
        // Compute U.x_(n+1)
        matA->UpperProduct(vecx, ilu, flux_Jacobian);
        RealFlow diag = matA->GetDiag(ilu);
        // Compute x_(n+1) = x*-U.x_(n+1)/D
        IntType idx = ilu * nVar;
        for(IntType iVar=0; iVar < nVar; iVar++)
        {
            vecx[idx+iVar] -= flux_Jacobian[iVar]/diag;
        }
    }

    mfmem::sdel_array_1D(flux_Jacobian);
}

#ifdef USING_PETSC
PetscErrorCode LinSysSolver::Solve( Vec & vecb, Vec & vecx )
{
    return KSPSolve(ksp, vecb, vecx);
}

void LinSysSolver::SetPC( PetscErrorCode (*userPC)(PC,Vec,Vec))
{
    PC pc;
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCSHELL);
    PCShellSetApply(pc,userPC);
}

void LinSysSolver::SetPC( IntType pctype )
{
    PC pc;
    KSPGetPC(ksp, &pc);
    switch (pctype)
    {
    case LinSysPC::TYPE::NONE:
        break;
    case LinSysPC::TYPE::JACOBI:
        PCSetType(pc, PCJACOBI);
        break;
    case LinSysPC::TYPE::BJACOBI:
        PCSetType(pc, PCBJACOBI);
        break;
    case LinSysPC::TYPE::ILU:
        PCSetType(pc, PCILU);
        break;
    case LinSysPC::TYPE::ASM:
        PCSetType(pc, PCASM);
    default:
        break;
    }
    ///pc can change from run-time
    //PCSetFromOptions(pc);
    //PCSetUp(pc);
}

void LinSysSolver::SetPCContext( void *ctx )
{
    PC pc;
    KSPGetPC(ksp,&pc);
    PCShellSetContext(pc, ctx);
}

void LinSysSolver::SetTolerances( RealFlow tol, IntType maxits )
{
    KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,maxits);
}

void LinSysSolver::SetRestart(IntType kspan)
{
    KSPGMRESSetRestart(ksp, kspan);
}
#endif

}