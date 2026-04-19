/*******************************************************************************
                       High Re Numerical Wind Tunnel
                     Computational Aerodynamics Institute
                   China Aerodynamics Research&Development Center                   
*******************************************************************************/

/*******************************************************************************
* FILE:        grid_base.inl                                            
* PURPOSE:     inline functions of class Grid
* AUTHOR(S):   
*******************************************************************************/

inline IntType Grid::GetZone() const 
{ 
    return zn;
}

inline void Grid::SetZone(const IntType znin) 
{ 
    zn = znin;
}

inline void Grid::SetNTNode(const IntType ntn) 
{ 
    nTNode = ntn;
}

inline void Grid::SetX(RealGeom *xin) 
{
    mfmem::sSet(x,xin);
}

inline void Grid::SetY(RealGeom *yin) 
{
    mfmem::sSet(y,yin);
}

inline void  Grid::SetZ(RealGeom *zin) 
{
    mfmem::sSet(z,zin);
}

inline RealGeom* Grid::GetX() const 
{ 
    return x;
}

inline RealGeom* Grid::GetY() const 
{ 
    return y;
}

inline RealGeom* Grid::GetZ() const 
{ 
    return z;
}

inline IntType Grid::GetNTNode() const 
{ 
    return nTNode;
}

inline void Grid::CopyDataFrom(DataSafe *in) 
{
    gPara = in;
}

inline void Grid::CopyFieldFrom(DataStore *in) 
{
    gField = in;
}

inline void Grid::UpdateDataPtr(void *data,IntType type,IntType size,const ShortString name) 
{
    gField->UpdateDataStore(data,type,size,name); 
}

inline void *Grid::GetDataPtr(IntType type, IntType size, const ShortString name) const 
{
    return gField->GetDataPtrByName(type,size,name); 
}

inline void Grid::DeleteDataPtr(const ShortString name) 
{
    gField->DeleteDataByName(name); 
}

inline void Grid::UpdateData(void *data, IntType type, IntType size, const ShortString name)
{
    gPara->UpdateDataSafe(data,type,size,name);
}

inline void Grid::GetData(void *data, IntType type, IntType size, const ShortString name) const
{
    gPara->GetDataByName(data,type,size,name);
}

inline void Grid::GetData(void *data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    gPara->GetDataByName(data,type,size,name,messageOn);
}

