//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   boundary_condition.h
/// \brief  BCond Object
/// \author 
/// \date   
/// \copyright  C.All rights reserved. 2010-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_BOUNDARY_CONDITION_H
#define MFL_BOUNDARY_CONDITION_H

#include<string.h>
#include "number_type.h"
#include "constant.h"
#include "data_pool.h"


namespace mflow
{

class BCRecord
{

private:
    IntType     type_, patch_id_; // the bc key(or bc type), patch id (base 1)
    ShortString type_symbols_;    // the type symbol, size less than MAX_SHORT_STRING
    String      patch_name_;      // the type string, patch name
    DataSafe    bcData_;          // use this class to store bc data

public:

    /// \brief Set the patch type string defined in MFlow,
    /// such as "wall", "symm", "far_field", et al.
    void SetTypeSymbol(const ShortString type_symbols);

    /// \brief Set the name of this boundary patch
    void SetPatchName(const String patch_name);

    /// \brief Set the patch type of an integer 
    void SetType(IntType type);

    /// \brief Set the id of this boundary patch
    void SetPatchID(IntType id);

    /// \brief Get the bc type string defined in MFlow,
    /// such as "wall", "symm", "far_field", et al. 
    const ShortString *GetTypeSymbol() const;

    /// \brief Get the patch name
    const String *GetPatchName() const;

    /// \brief Get the patch type of an integer 
    IntType GetType() const;

    /// \brief Get the id of this boundary patch
    IntType GetPatchID() const;

    /// \brief Get the data pointer
    const DataSafe * GetDataPointer(void) const;

    /// \brief Get the data reference
    DataSafe & GetDataReference(void);

    /// \brief Update one data in data pool binded with this patch
    void UpdateData(void *data, IntType typein, IntType size, const ShortString name);

    /// \brief Get one data in data pool binded with this patch
    void GetBCVar(void *value, IntType type, const ShortString name);   
    void GetBCVar(void *value, IntType type, const ShortString name, IntType messageOn);

    /// \brief delete all data which are given with $ for this BCRecord.
    void EraseExtraData(void);

    /// \brief Deeply copy data from the other
    void CopyFrom(const BCRecord* other);

    // constructor
    BCRecord(const IntType type, const ShortString type_symbols, const String patch_name);
    BCRecord(const ShortString type_symbols, const String patch_name);
    explicit BCRecord(const ShortString type_symbols);
    explicit BCRecord(const IntType type);
    BCRecord();

    // destructor
   ~BCRecord();
};

// inline functions for class BCRecord

// Set the bc type with an string
inline void BCRecord::SetTypeSymbol(const ShortString type_symbols)
{
    strcpy(type_symbols_, type_symbols);
}

// Set the name of this boundary patch
inline void BCRecord::SetPatchName(const String patch_name)
{
    strcpy(patch_name_, patch_name);
}

// Set the patch type with an integer 
inline void BCRecord::SetType(IntType type)
{
    type_ = type;
};

// Set the id of this boundary patch
inline void BCRecord::SetPatchID(IntType id)
{
    patch_id_ = id;
}

// Get the bc type with an string defined in MFlow,
// such as "wall", "symm", "far_field", et al. 
inline const ShortString * BCRecord::GetTypeSymbol() const
{
    return &type_symbols_;
}

// Get the patch name
inline const String * BCRecord::GetPatchName() const
{
    return &patch_name_;
}

// Get the patch type with an integer 
inline IntType BCRecord::GetType() const
{
    return type_;
}

// Get the id of this boundary patch
inline IntType BCRecord::GetPatchID() const
{
    return patch_id_;
}

// Get the data pointer
inline const DataSafe * BCRecord::GetDataPointer(void) const
{
    return &bcData_;
}

// Get the data reference
inline DataSafe & BCRecord::GetDataReference(void)
{
    return bcData_;
}

inline BCRecord::BCRecord(const IntType type, const ShortString type_symbols, const String patch_name) : type_(type)
{
    strcpy(type_symbols_, type_symbols);
    strcpy(patch_name_, patch_name);
}

inline BCRecord::BCRecord(const ShortString type_symbols, const String patch_name) : type_(0)
{
    strcpy(type_symbols_, type_symbols);
    strcpy(patch_name_, patch_name);
}

inline BCRecord::BCRecord(const ShortString type_symbols) : type_(0) 
{
    strcpy(type_symbols_, type_symbols);
    patch_name_[0] = '\0';
}

inline BCRecord::BCRecord(const IntType type) : type_(type)
{
    type_symbols_[0] = '\0';
    patch_name_[0] = '\0';
}

inline BCRecord::BCRecord() : type_(0) 
{
    type_symbols_[0] = '\0';
    patch_name_[0] = '\0';
}

inline BCRecord::~BCRecord()
{
}

//
// Container to save several BCRecord
class BCond
{
    IntType    nBCRecord;
    BCRecord **bcRecord;     // the bc rec. each boun. face is assoc.

public:

    /// \brief constructor
    BCond();

    /// \brief Get the number of BCRecords
    IntType GetNoBCR() const;

    /// \brief Add a BCRecord
    void AddBCRecord(BCRecord *bcRecord);

    /// \brief Get the i-th BCRecord item
    BCRecord *GetBCRecord(IntType i) const;

    /// \brief Add or update boundary conditions from the other
    void CopyFrom(const BCond *other);

    /// \brief Return the BCRecord which patch id is patch_id, return NULL if not exist.
    BCRecord *FindBCRecord(const IntType patch_id);

    /// \brief Destructor
    ~BCond();
};

// inline functions for class BCond

inline BCond::BCond() : nBCRecord(0)
{
    bcRecord  = NULL;
}

inline IntType  BCond::GetNoBCR() const
{
    return nBCRecord;
}

inline BCRecord * BCond::GetBCRecord(IntType i) const
{
    return bcRecord[i];
}

} // ~namespace mflow
    
#endif //~MFL_BOUNDARY_CONDITION_H
