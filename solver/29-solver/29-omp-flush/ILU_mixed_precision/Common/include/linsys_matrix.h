#ifndef LINSYS_MATRIX_H
#define LINSYS_MATRIX_H
#include "number_type.h"
#include "linsys_vector.h"
#include <vector>
#include <map>
namespace mflow
{

class LinSysMatrix
{
public:
    enum TYPE
    {
        NONE, MATRIX_FREE, MATRIX_FREE_PETSC, BLOCK_MATRIX_PETSC, BLOCK_MATRIX_VIENNACL
    };
private:
    LinSysMatrix::TYPE type;
    IntType nVar;    /// number of vars solved in the equation
    IntType nElem;   /// total number of cells or elements in this processor
    IntType size;    /// the real matrix size
    /// The data member used for matrix-free type
    void * ctx;           /// ctx is a pointer to data needed by any user-defined matrix operations, the name is same with petsc,  user-defined application context
    RealFlow * diag;      /// diagonal data of the matrix, the system matrix just keep the pointer to the diag, do not relly store the data
#ifdef USING_PETSC
    Mat petscMatrix;      /// using petsc's matrix
#endif
#ifdef USING_VIENNACL
    std::vector<std::map<unsigned int, RealFlow> > viennaclMatrix;
#endif
    IntType Istart, Iend;
    IntType Bstart, Bend;
public:
    /// \brief Performs the product of i-th row of the upper part of a sparse matrix by a vector.
    /// \param [in] vec - Vector to be multiplied by the upper part of the sparse matrix A.
    /// \param [in] row - Row of the matrix to be multiplied by vector vec.
    /// \param [out] prod - Result of the product U(A)*vec
    void UpperProduct(const LinSysVector &vec, IntType row, RealFlow * prod);

    /// \brief Performs the product of i-th row of the lower part of a sparse matrix by a vector.
    /// \param [in] vec - Vector to be multiplied by the lower part of the sparse matrix A.
    /// \param [in] row - Row of the matrix to be multiplied by vector vec.
    /// \param [out] prod - Result of the product L(A)*vec
    void LowerProduct(const LinSysVector &vec, IntType row, RealFlow * prod);

    /// \brief set the pointer to the function of user-defined upper product
    int SetUpperProduct( void (*pUpperProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod) );

    /// \brief set the pointer to the function of user-defined lower product
    int SetLowerProduct( void (*pLowerProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod) );

    /// get the 
    IntType getSize() const { return size;}

    /// \brief return the userd-defiend data pointer
    void * GetCTX() const;

    void SetDiag(RealFlow * diag);

    RealFlow  GetDiag(IntType iElem) const;

    IntType GetNVar() const;

    IntType GetNElem() const;

    void SetCtx(void * ctx);

    LinSysMatrix::TYPE GetType() const { return type;}

    void SetValue(IntType iPos, IntType jPos, RealFlow val);

    void SetBlockValue(IntType iBlock, IntType jBlock, RealFlow *val);

    void AssemblyMatPetsc();

    void PreallocationCOO(IntType count, IntType *oor, IntType *ooc);

    void SetValuesCOO(RealFlow *v);

#ifdef USING_PETSC
    Mat GetMatPetsc() const { return petscMatrix;}

    int SetMatOpMultPetsc( void (*userMatMult)(void));
#endif

#ifdef USING_VIENNACL
    std::vector<std::map<unsigned int, RealFlow> >  & GetMatViennaCL() { return viennaclMatrix;}
#endif

    IntType GetIstart() const;

    IntType GetBstart() const;

    IntType GetBend() const;

public:
    /// constructor for the matrix-free type matrix
    LinSysMatrix(IntType nElem, IntType nVar, void *ctx, LinSysMatrix::TYPE type);

    LinSysMatrix(IntType nElem, IntType nVar, IntType * d_nnz, IntType *o_nnz, LinSysMatrix::TYPE type);

    ~LinSysMatrix();
private:

    /// pointer to the user-defined upperProduct
    void (*pUpperProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod);

    /// pointer to the user-defined lowerProduct
    void (*pLowerProduct)(const LinSysMatrix & mat, const LinSysVector &vec, IntType row, RealFlow * prod);
};

inline void * LinSysMatrix::GetCTX() const
{
    return ctx;
}

inline void LinSysMatrix::SetDiag(RealFlow * diag)
{
    this->diag = diag;
}

inline RealFlow  LinSysMatrix::GetDiag(IntType iElem) const
{
    return diag[iElem];  ///maybe its reordered
}

inline IntType LinSysMatrix::GetNElem() const
{
    return nElem;
}

inline IntType LinSysMatrix::GetNVar() const
{
    return nVar;
}

inline IntType LinSysMatrix::GetIstart() const
{
    return Istart;
}

inline IntType LinSysMatrix::GetBstart() const
{
    return Bstart;
}

inline IntType LinSysMatrix::GetBend() const
{
    return Bend;
}

}
#endif