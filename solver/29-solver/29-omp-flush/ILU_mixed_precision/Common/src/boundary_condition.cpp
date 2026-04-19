//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   boundary_condition.cpp
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

// direct head file
#include "boundary_condition.h"

// build-in head files
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
using namespace std;

// other user defined head file
#include "number_type.h"
#include "memory_util.h"

namespace mflow
{

BCond::~BCond()
{
    IntType i;
    for(i=0; i<nBCRecord; i++) {
        mfmem::sdel_object(bcRecord[i]);
    }
    mfmem::sdel_array_1D(bcRecord);
}

void BCond::AddBCRecord(BCRecord *bcRecordin)
{
    if(nBCRecord == 0) {
        mfmem::snew_array_1D(bcRecord,1,dmrfl);
        bcRecord[nBCRecord++] = bcRecordin;
    } else {
        IntType i;
        BCRecord **bcRecordt = bcRecord;
        bcRecord = NULL;
        mfmem::snew_array_1D(bcRecord,nBCRecord+1,dmrfl);
        for(i=0; i<nBCRecord; i++) bcRecord[i] = bcRecordt[i];
        bcRecord[nBCRecord++] = bcRecordin;
        mfmem::sdel_array_1D(bcRecordt);
    }
}

// Copy boundary conditions from the other
void BCond::CopyFrom(const BCond *other)
{
    IntType nbc = other->GetNoBCR();
    for (IntType ibc = 0; ibc < nbc; ++ibc)
    {
        BCRecord *bcr_other = other->GetBCRecord(ibc);
        BCRecord *bcr_this  = this->FindBCRecord(bcr_other->GetPatchID());

        if (bcr_this != NULL)
        {
            bcr_this->CopyFrom(bcr_other);
        }
        else
        {
            BCRecord *bcr = NULL;
            mfmem::snew_object(bcr, dmrfl);
            bcr->CopyFrom(other->GetBCRecord(ibc));
            this->AddBCRecord(bcr);
        }        
    }
}


/// Return the BCRecord which patch id is patch_id, return NULL if not exist.
BCRecord *BCond::FindBCRecord(const IntType patch_id)
{
    for (IntType ibc = 0; ibc < this->GetNoBCR(); ++ibc)
    {
        BCRecord * current_bcr = this->GetBCRecord(ibc);
        if (current_bcr->GetPatchID() == patch_id)
        {
            return current_bcr;
        }
    }

    return NULL;
}

void BCRecord::UpdateData(void *data, IntType typein, IntType size, const ShortString name)
{
    bcData_.UpdateDataSafe(data,typein,size,name);
}
void BCRecord::GetBCVar(void *data, IntType type, const ShortString name, IntType messageOn)
{
    bcData_.GetDataByName(data,type,1,name,messageOn);
}

void BCRecord::GetBCVar(void *data, IntType type, const ShortString name)
{
    bcData_.GetDataByName(data,type,1,name);
}

// delete extra data
void BCRecord::EraseExtraData(void)
{
    bcData_.DeleteAllData();
}

/// \brief Deeply copy data from the other
void BCRecord::CopyFrom(const BCRecord* other)
{
    this->SetType(other->GetType());
    this->SetPatchID(other->GetPatchID());
    this->SetTypeSymbol(*(other->GetTypeSymbol()));
    this->SetPatchName(*(other->GetPatchName()));
    this->EraseExtraData();
    this->bcData_.CopyDataFrom(other->GetDataPointer());
}

} // ~namespace mflow
