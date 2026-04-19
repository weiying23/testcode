#ifndef LINSYS_PRECONDITIONER_H
#define LINSYS_PRECONDITIONER_H
namespace mflow
{

class LinSysPC
{
public:
    enum TYPE
    {
        NONE, LUSGS, JACOBI, BJACOBI, ILU, ASM, LIBLUSGS, Chow_Patel_ILU0, BLOCK_ILU
    };
};

}
#endif