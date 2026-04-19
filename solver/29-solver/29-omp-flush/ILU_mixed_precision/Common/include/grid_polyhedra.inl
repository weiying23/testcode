/*******************************************************************************
                       High Re Numerical Wind Tunnel
                     Computational Aerodynamics Institute
                   China Aerodynamics Research&Development Center                   
*******************************************************************************/

/*******************************************************************************
* FILE:        grid_polyhedra.inl                                            
* PURPOSE:     inline functions of class PolyGrid
* AUTHOR(S):   
*******************************************************************************/

#ifdef DC0

inline void PolyGrid::SetuTaskTree(uTaskTree *in) 
{
    mfmem::sSet(uTaskTreeRoot,in);
}

inline uTaskTree *PolyGrid::GetuTaskTree() const
{
    return uTaskTreeRoot;
}
#endif

inline void  PolyGrid::SetNTFace(const IntType in) 
{
    nTFace = in;
}

inline void PolyGrid::SetNTCell(const IntType in) 
{
    nTCell = in;
}

inline void PolyGrid::SetNBFace(const IntType in) 
{
    nBFace = in;
}

inline void PolyGrid::SetNIFace(const IntType in) 
{
    nIFace = in;
}

inline void PolyGrid::SetNINode(const IntType in) 
{
    nINode = in;
}

inline IntType PolyGrid::GetNTFace() const 
{
    return nTFace;
}

inline IntType PolyGrid::GetNTCell() const 
{
    return nTCell;
}

inline IntType PolyGrid::GetNBFace() const 
{
    return nBFace;
}

inline IntType PolyGrid::GetNIFace() const 
{
    return nIFace;
}

inline IntType PolyGrid::GetNINode() const 
{
    return nINode;
}

inline void PolyGrid::SetnNPF(IntType *in) 
{
    mfmem::sSet(nNPF,in);
}

inline void PolyGrid::SetnFPC(IntType *in) 
{
    mfmem::sSet(nFPC,in);
}

inline void PolyGrid::SetnNPC(IntType *in) 
{
    mfmem::sSet(nNPC,in);
}

inline void PolyGrid::Setf2n(IntType *in) 
{
    mfmem::sSet(f2n,in);
}

inline void PolyGrid::Setf2c(IntType *in) 
{
    mfmem::sSet(f2c,in);
}

inline RealGeom  *PolyGrid::GetXcc() const
{
    return xcc;
}

inline RealGeom  *PolyGrid::GetYcc() const
{
    return ycc;
}

inline RealGeom  *PolyGrid::GetZcc() const
{
    return zcc;
}

// Get and set cell volume     
inline RealGeom * PolyGrid::GetCellVol() const
{
    return vol;
}

inline RealGeom  *PolyGrid::GetXfc() const
{
    return xfc;
}

inline RealGeom  *PolyGrid::GetYfc() const
{
    return yfc;
}

inline RealGeom  *PolyGrid::GetZfc() const
{
    return zfc;
}

inline RealGeom  *PolyGrid::GetXfn() const
{
    return xfn;
}

inline RealGeom  *PolyGrid::GetYfn() const
{
    return yfn;
}

inline RealGeom  *PolyGrid::GetZfn() const
{
    return zfn;
}

// Get and set face aera     
inline RealGeom * PolyGrid::GetFaceArea() const
{
    return area;
}

inline void PolyGrid::SetnCPC(IntType *in) 
{
    mfmem::sSet(nCPC,in);
}

inline void PolyGrid::Setc2c(IntType **in) 
{
    mfmem::sSet(c2c,in);
}

inline void PolyGrid::SetC2N(IntType **in) 
{
    mfmem::sSet(C2N,in);
}

inline void PolyGrid::SetC2F(IntType **in) 
{
    mfmem::sSet(C2F,in);
}

// F2N is special and just a reference to f2n, so use 1D array operator, tangj 
inline void PolyGrid::SetF2N(IntType **in) 
{
    if((in!=NULL)&&(F2N!=in)){ 
        mfmem::sdel_array_1D(F2N); 
    }
    F2N = in;
}  

inline IntType *PolyGrid::GetnNPF() const 
{
    return nNPF;
}

inline IntType * PolyGrid::GetnFPC() const 
{
    return nFPC;
}

inline IntType * PolyGrid::GetnNPC() const 
{
    return nNPC;
}

inline IntType * PolyGrid::Getf2n() const 
{
    return f2n;
}

inline IntType * PolyGrid::Getf2c() const 
{
    return f2c;
}


inline IntType *PolyGrid::GetnCPC() const 
{
    return nCPC;
}

//node to cell relationships:
inline IntType **PolyGrid::GetN2C() const
{
    return N2C;
}

inline IntType *PolyGrid::GetnCPN() const
{
    return nCPN;
}

inline void PolyGrid::SetN2C(IntType** in)
{
    mfmem::sSet(N2C, in);
}

inline void PolyGrid::SetnCPN(IntType* in)
{
    mfmem::sSet(nCPN, in);
}

inline IntType **PolyGrid::Getc2c() const 
{
    return c2c;
}

inline IntType **PolyGrid::GetC2N() const 
{
    return C2N;
}

inline IntType **PolyGrid::GetC2F() const 
{
    return C2F;
}

inline IntType **PolyGrid::GetF2N() const 
{
    return F2N;
}

inline void PolyGrid::SetNumberOfFaceNeighbors(const IntType n)
{
    nNeighbor = n;
}

inline IntType PolyGrid::GetNumberOfFaceNeighbors() const
{
    return nNeighbor;
}
inline void PolyGrid::SetNumberOfNodeNeighbors(const IntType n)
{
    nNeighborN = n;
}

inline IntType PolyGrid::GetNumberOfNodeNeighbors() const
{
    return nNeighborN;
}

inline void PolyGrid::SetFaceNeighborZones(IntType *fnz)
{
    mfmem::sSet(nb, fnz);
}

inline IntType * PolyGrid::GetFaceNeighborZones() const
{
    return nb;
}

inline void PolyGrid::SetNodeNeighborZones(IntType *nnz)
{
    mfmem::sSet(nbN, nnz);
}

inline IntType * PolyGrid::GetNodeNeighborZones() const
{
    return nbN;
}

inline void PolyGrid::SetNeighborGrids(PolyGrid **grids)
{
    // This is an object array, we only delete the array
    // and do not delete the objects passed in the array.
    if((nbg!=NULL) && (nbg!=grids))
    { 
        mfmem::sdel_array_1D(nbg); 
    }
    nbg = grids;
}

inline void PolyGrid::SetnbZ(IntType *in) 
{
    mfmem::sSet(nbZ,in);
}

inline IntType *PolyGrid::GetnbZ() const 
{
    return nbZ;
}

inline void PolyGrid::SetnbBF(IntType *in) 
{
    mfmem::sSet(nbBF,in);
}

inline IntType *PolyGrid::GetnbBF() const 
{
    return nbBF;
}

inline void PolyGrid::SetnbSN(IntType *in) 
{
    mfmem::sSet(nbSN,in);
}

inline IntType *PolyGrid::GetnbSN() const 
{
    return nbSN;
}

inline void PolyGrid::SetnbZN(IntType *in) 
{
    mfmem::sSet(nbZN,in);
}

inline IntType *PolyGrid::GetnbZN() const 
{
    return nbZN;
}

inline void PolyGrid::SetnbRN(IntType *in) 
{
    mfmem::sSet(nbRN,in);
}

inline IntType *PolyGrid::GetnbRN() const 
{
    return nbRN;
}

inline IntType PolyGrid::GetLevel() const 
{
    return level;
}

inline void PolyGrid::SetLevel(IntType lin)  
{ 
    level = lin;
}

inline void PolyGrid::Setbcr(BCRecord **bcrs) 
{
    // This is an object array, we only delete the array
    // and do not delete the objects passed in the array.
    if((bcr != NULL) && (bcr != bcrs))
    { 
        mfmem::sdel_array_1D(bcr); 
    }
    bcr = bcrs;
}

inline BCRecord **PolyGrid::Getbcr() const 
{
    return bcr;
}

inline void PolyGrid::SetVolAvg(RealGeom in) 
{
    VolAvg = in;
}

inline RealGeom PolyGrid::GetVolAvg() const 
{
    return VolAvg;
}

//--------------------------------------------------------- 
inline RealGeom *PolyGrid::GetGridQualityFaceCentroidSkewness(void) const
{
    return facecentroidskewness;
}

inline IntType *PolyGrid::GetGridQualityCellWallNumber(void) const
{
    return cellwallnumber;
}
// Flow reconstruction from cell to node
//---------------------------------------------------------

inline void PolyGrid::SetNodeType(IntType *node_type)
{
    mfmem::sSet(Nmark, node_type);
}

inline IntType *PolyGrid::GetNodeType(void) const
{
    return Nmark;
}

inline void PolyGrid::SetWeightNodeDist(RealGeom *weight)
{
    mfmem::sSet(WeightNodeDist, weight);
}

inline RealGeom *PolyGrid::GetWeightNodeDist(void) const
{
    return WeightNodeDist;
}

inline void PolyGrid::SetWeightNodeC2N(RealGeom **weight_c2n)
{
    mfmem::sSet(WeightNodeC2N, weight_c2n);
}

inline RealGeom **PolyGrid::GetWeightNodeC2N(void) const
{
    return WeightNodeC2N;
}

inline void PolyGrid::SetWeightNodeN2C(RealGeom** weight_n2c)
{
    mfmem::sSet(WeightNodeN2C, weight_n2c);
}

inline RealGeom** PolyGrid::GetWeightNodeN2C(void) const
{
    return WeightNodeN2C;
}

inline void PolyGrid::SetWeightNodeBFace2C(RealGeom** WeightNodebFace2c)
{
    mfmem::sSet(WeightNodeBFace2C, WeightNodebFace2c);
}

inline RealGeom** PolyGrid::GetWeightNodeBFace2C(void) const
{
    return WeightNodeBFace2C;
}

//--------------------------------------------------------- 
// moving grid
//---------------------------------------------------------
inline RealGeom * PolyGrid::GetFaceNormalVelocity(void) const
{
    return vgn;
}

inline RealGeom * PolyGrid::GetBoundaryFaceVelocityX(void) const
{
    return BFacevgx;
}

inline RealGeom * PolyGrid::GetBoundaryFaceVelocityY(void) const
{
    return BFacevgy;
}

inline RealGeom * PolyGrid::GetBoundaryFaceVelocityZ(void) const
{
    return BFacevgz;
}


