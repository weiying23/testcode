//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   data_pool.cpp
/// \brief  Data pool to save parameters or flow field
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
#include "data_pool.h"

// C++ build-in head files
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

// other user defined head file
#include "number_type.h"
#include "system_base_functions.h"
#include "memory_util.h"

namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 10401  // define file id

class DataNode 
{
public:
    ShortString name;
    IntType     type;
    IntType     size;
    void       *data;          // Be careful, the implementation just stores a
    DataNode   *next;          // pointer.  The allocation is done by user outside
                            // this scope.
    DataNode(){ data = NULL; next = NULL; name[0] = '\0'; }
   ~DataNode(){ mfmem::sdel_void_array_1D(data); }
};


/*******************************************************************************
*                                                                              *
*******************************************************************************/
DataSafe::DataSafe() : nData(0)
{
    top = NULL;
    mfmem::snew_object(top,dmrfl);
}


DataSafe::~DataSafe()
{
    DeleteAllData();
    mfmem::sdel_object(top);
}


/************************************************************************
*  Given the type, return the data length in bytes
************************************************************************/
size_t SizeOfType(IntType type)
{
    size_t len = 999;  //999 avoiding to random value 

    if(type == INT) {
        len = sizeof(IntType);
    } else if(type == REAL_FLOW) {
        len = sizeof(RealFlow);
    } else if(type == REAL_GEOM) {
        len = sizeof(RealGeom);
    } else if(type == STRING) {
        len = MAX_STRING;
    } else if(type == FLOAT) {
        len = sizeof(float);
    } else if(type == DOUBLE) {
        len = sizeof(double);
    } else if(type == CHAR) {
        len = sizeof(char);
    } else if(type == LONG) {
        len = sizeof(long);
    } else {
        fprintf(stdout,"unknown type in SizeOfType");
    }
    return len;
}


void DataSafe::UpdateDataSafe(void *data, IntType type, IntType size, const ShortString name)
{
    DataNode *p, *pt = NULL;
    size_t  len;

    p = top;

    while(p->next) {
        p = p->next;
      if(strcmp(p->name, name) == 0) {

          // Bytes of new data
          len = size*SizeOfType(type);
          
          if(p->type != type || p->size != size) {
              fprintf(stdout, "Update Warning!!!! Size or type of \"%s\" mis-match\n",name);
              if(p->type != type){ //exit
                  mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
              }else{ //replaced with new data
                  p->size = size;
                  
                  // free old memory of this DataS
                  mfmem::sdel_void_array_1D(p->data);                  
                  
                  // allocate new memory
                  char *pc = NULL;
                  mfmem::snew_array_1D(pc, len, dmrfl);
                  
                  p->data = (void *)pc;
              }              
          }   

          memcpy(p->data, data, len);
          return;
        }
    }

    // add it
    nData++;
    mfmem::snew_object(pt, dmrfl);
    strcpy(pt->name, name);
    pt->type = type;
    pt->size = size;

    len = size*SizeOfType(type);

    char *pc = NULL;
    mfmem::snew_array_1D(pc, len, dmrfl);

    pt->data = (void *)pc;
    memcpy(pt->data, data, len);

    p->next = pt;
}


void DataSafe::GetDataByName(void * data, IntType type, IntType size, const ShortString name) const
{
    DataNode *p;
    IntType  len;

    p = top;

    while(p->next) {
        p = p->next;
        if(strcmp(p->name, name) == 0) {
            if(p->type != type || p->size != size) {
                fprintf(stdout, "GetData Warning!!!! Size or type mis-match in GetDataByName\n");
                printf("Data of name %s not found\n", name);  
            }

            len = static_cast<IntType> (p->size*SizeOfType(p->type));
            memcpy(data, p->data, len);

            return ;
        }
    }
    printf("Data of name %s not found\n", name);  
}


void DataSafe::GetDataByName(void * data, IntType type, IntType size, const ShortString name, IntType messageOn) const
{
    DataNode *p;
    IntType  len;

    p = top;

    while(p->next) {
        p = p->next;
        if(strcmp(p->name, name) == 0) {
            if(p->type != type || p->size != size) {
                fprintf(stdout, "GetData Warning!!!! Size or type mis-match in GetDataByName\n");
                printf("Data of name %s not found\n", name);  
            }

            len = static_cast<IntType> (p->size*SizeOfType(p->type));
            memcpy(data, p->data, len);

            return ;
        }
    }
    if(messageOn) {
        printf("Data of name %s not found\n", name);
    }  
}


void DataSafe::DeleteDataByName(const ShortString name)
{
    DataNode *p = top;

    while(p->next) {
        DataNode *last = p;
        p = p->next;
        if(strcmp(p->name, name) == 0) {
            last->next = p->next;    
            mfmem::sdel_object(p);
            nData--;

            break;
        }
    }
}


void DataSafe::DeleteAllData()
{
    DataNode *p = top->next;

    while(p) {
        DataNode *pc = p;
        p = p->next;
        DeleteDataByName(pc->name);
    }
}


void DataSafe::ListAllData() const
{
    DataNode *p;
    IntType  *ip,i;
    RealFlow *rf;
    RealGeom *rg;
    float    *fl;
    double   *db;
    String   *st;

    p = top->next;

    IntType count=0;

    fprintf(stdout,"\n Parameters are listed as bellow:\n");
    while(p) {
        fprintf(stdout,"item %3d: name %15s, size %d, ", ++count, p->name, p->size);

        if(p->type == INT) {
            fprintf(stdout, "type  int, ");
        } else if(p->type == REAL_GEOM) {
            fprintf(stdout, "type real, ");
        } else if(p->type == REAL_FLOW) {
            fprintf(stdout, "type real, ");
        } else if(p->type == FLOAT) {
            fprintf(stdout, "type real, ");
        } else if(p->type == DOUBLE) {
            fprintf(stdout, "type real, ");
        } else if(p->type == STRING) {
            fprintf(stdout, "type  str, ");
        } else { 
            fprintf(stdout, "type unkn, ");
        }
        
        for(i=0; i<p->size; i++) {
            if(p->type == INT) {
                ip = (IntType *) p->data;
                fprintf(stdout,"%d ", ip[i]);
            } else if(p->type == REAL_GEOM) {
                rg = (RealGeom *) p->data;
                fprintf(stdout,"%.7e ",rg[i]); 
            } else if(p->type == REAL_FLOW) {
                rf = (RealFlow *) p->data;
                fprintf(stdout,"%.7e ",rf[i]); 
            } else if(p->type == FLOAT) {
                fl = (float *) p->data;
                fprintf(stdout,"%.7e ",fl[i]); 
            } else if(p->type == DOUBLE) {
                db = (double *) p->data;
                fprintf(stdout,"%.7e ",db[i]); 
            } else if(p->type == STRING) {
                st = (String *) p->data;
                fprintf(stdout,"%s ",st[i]);
            } else { 
                fprintf(stdout,"unknown type");
            }
        }

        fprintf(stdout,"\n");
        p = p->next;
    }
}


// Add or update all the data of DataSafe object src
void DataSafe::CopyDataFrom(const DataSafe *src)
{
    // node 'top' has no real data
    DataNode *p = src->top->next;

    while(p != NULL)
    {
        this->UpdateDataSafe(p->data, p->type, p->size, p->name);
        p = p->next;
    }
}


DataStore::DataStore() : nData(0)
{
    top = NULL;
    mfmem::snew_object(top,dmrfl);
}


DataStore::~DataStore()
{
    DeleteAllData();
    mfmem::sdel_object(top);
}


void DataStore::UpdateDataStore(void *data, IntType type, IntType size, const ShortString name)
{
    DataNode *p, *pt = NULL;

    p = top;

    while(p->next) {
        p = p->next;
        if(strcmp(p->name, name) == 0) {
            if(p->type != type || p->size != size) {
                fprintf(stdout,"Update Warning!!!! Size or type of \"%s\" mis-match\n", name);
            }

            if(p->data != data) {    
                mfmem::sdel_void_array_1D(p->data);
            }

            p->data = data;   // note that only the pointer is stored
            return;           // A more robust way is to allocate memory.
                              // and really store the data
        }
    }

    // add it

    nData++;

    mfmem::snew_object(pt,dmrfl);
    strcpy(pt->name, name);
    pt->type = type;
    pt->size = size;
    pt->data = data;

    p->next = pt;
}


void *DataStore::GetDataPtrByName(IntType type, IntType size, const ShortString name) const
{
    void     *data=0;
    DataNode *p;

    p = top;

    while(p->next) {
        p = p->next;
        if(strcmp(p->name, name) == 0) {
            if(p->type != type || p->size != size) {
                fprintf(stdout, "GetData Warning!!!! Size or type mis-match for variable %s\n", name);
                fprintf(stdout, "Existent size and type: %d %d\n", p->size, p->type);
                fprintf(stdout, "Wanted   size and type: %d %d\n", size, type);
            }

            data = p->data;
            break;
        }
    }
    return data;
}


void DataStore::DeleteDataByName(const ShortString name)
{
    DataNode *p = top;
    while(p->next) {
        DataNode *last = p;
        p = p->next;
        if(strcmp(p->name, name) == 0){
            last->next = p->next;
            mfmem::sdel_object(p);
            nData--;
            break;
        }
    }
}


void DataStore::DeleteAllData()
{
    DataNode *p = top->next;

    while(p) {
        DataNode *pc = p;
        p = p->next;
        DeleteDataByName(pc->name);
    }
}


#undef CPP_FILD_ID  // clear out file id
} //~namespace mflow
