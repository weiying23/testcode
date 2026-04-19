//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   system_base_functions.cpp
/// \brief  Functions related to system
/// \author tangj
/// \date   2020.03.23
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

// direct head file
#include "system_base_functions.h"

// standard head files
#include <stdlib.h>  // std::exit()
#include <iostream>  //

// other user defined head files
#include "io_log.h"

// head files relying on condition-compiling
#ifdef MPICH
#include "mpi.h"
#endif

// for Creatfolder
#ifndef _LINUX_
#include <io.h>  // access()
#else
#include <unistd.h> // access()
#include <fcntl.h>
#include <cstdlib>  // system()
#endif

namespace mflow
{

#ifdef MPICH
extern int numprocs;
extern int myZone;
extern MPI_Comm GridComm;  //for each grid, tangj
#endif


// exit program and output an error flag
void mflow_exit(const int error_flag)
{
    mflog::log.set_each_grid_out();
    mflog::log << "Error code: " << error_flag << std::endl;
#ifdef MPICH
    MPI_Abort(MPI_COMM_WORLD, error_flag);
#else
    std::exit(error_flag);
#endif
}


/******************************************************************************\
|  在当前目录(./)创建名字为fname的文件夹
\******************************************************************************/
void CreatFolder(const std::string &fname)
{
    std::string FolderDir;
    FolderDir = "./";
    FolderDir += fname;
    FolderDir += "/";

    // disable warning #4996 with visual studio(use _access instead of access)
#pragma warning(disable: 4996)
    // folder exist, return
    if ( access(FolderDir.c_str(), 0) == 0) return;
#pragma warning(default: 4996)

    // not exist, creat it.
#ifdef _LINUX_
    std::string str;
    str="mkdir \"";
    str += FolderDir;
    str += "\"";
    system(str.c_str());
#else
    std::string str;
    str="MKDIR  \"";
    str += FolderDir;
    str += "\"";
    system(str.c_str());
#endif
}


void CreatFolder_OneProcessor(const std::string &fname, const int porc)
{
#ifdef MPICH
    if(myZone == porc) {
#endif
        CreatFolder(fname);

#ifdef MPICH
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

// check whether a file is readable.
bool CheckFileReadable(const std::string & filename)
{
    std::ifstream in(filename.c_str(), std::ifstream::in);
    if (in.fail()) return false;
    in.close();
    return true;
}

} // namespace mflow

