#ifdef USING_VIENNACL
#ifndef FS_CUDA
#include "linsys_solver_viennacl.h"
#include "linsys_preconditioner.h"
#include "io_log.h"
#include "system_base_functions.h"

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"

#ifdef GMRESDEBUG
#include <ctime>
#include <fstream>
#if !(defined(Windows_NT) )
#include <sys/time.h>
#endif
#endif

namespace mflow
{
namespace viennaclData
{
    RealFlow tol;
    IntType maxits;
    IntType kspan;  
    IntType pcType;
    IntType chow_patel_ilu_sweeps;
    IntType chow_patel_ilu_jacobi_iters;
    IntType block_ilu_size;
    bool ilu_use_level_scheduling;
}

void SetViennaclGMRESPara(RealFlow tol, IntType maxits, IntType kspan, IntType pcType)
{
    viennaclData::tol = tol;
    viennaclData::maxits = maxits;
    viennaclData::kspan = kspan;
    viennaclData::pcType = pcType;
}

void SetViennaclILULevelScheduling(IntType flag)
{
    if(flag)
    {
        viennaclData::ilu_use_level_scheduling = true;
    }
    else
    {
        viennaclData::ilu_use_level_scheduling = false;
    }
}

void SetViennaclChowPatelILUPara(IntType sweeps, IntType jacobi_iters)
{
    viennaclData::chow_patel_ilu_sweeps = sweeps;
    viennaclData::chow_patel_ilu_jacobi_iters = jacobi_iters;
}

void SetViennaclBlockILUSize(IntType size)
{
    viennaclData::block_ilu_size = size;
}

void viennaclGMRESSolve(std::vector<std::map<unsigned int, RealFlow> > & stl_A, std::vector<RealFlow> & stl_b, std::vector<RealFlow> & stl_x )
{
    /// copy matrix to device:
    viennacl::compressed_matrix<RealFlow> A;
    viennacl::copy(stl_A, A);

    // copy vector to device:
    viennacl::vector<RealFlow> b(A.size2());
    copy(stl_b.begin(), stl_b.end(), b.begin());

    // solve:
    viennacl::linalg::gmres_tag my_gmres_tag(viennaclData::tol, viennaclData::maxits, viennaclData::kspan);
    viennacl::linalg::gmres_solver<viennacl::vector<RealFlow> > my_gmres_solver(my_gmres_tag);

    viennacl::vector<RealFlow> x;

    switch (viennaclData::pcType)
    {
    case LinSysPC::TYPE::JACOBI:
        {
            viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<RealFlow> > jacobi(A, viennacl::linalg::jacobi_tag());
            x = my_gmres_solver(A, b, jacobi);
        }
        break;

    case LinSysPC::TYPE::ILU:
        {
#ifdef GMRESDEBUG
            RealFlow time_gmres_pc;
#if !(defined(Windows_NT))
            timeval    t_tmp;
            gettimeofday(&t_tmp, NULL);
            time_gmres_pc = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000;
#else
            time_gmres_pc = ((double)clock()) / CLOCKS_PER_SEC;
#endif
#endif
            viennacl::linalg::ilu0_tag ilu0_config;
            if(viennaclData::ilu_use_level_scheduling)
            {
                ilu0_config.use_level_scheduling(true);
            }
            viennacl::linalg::ilu0_precond< viennacl::compressed_matrix<RealFlow>  > ilu0(A, ilu0_config);

#ifdef GMRESDEBUG
#if !(defined(Windows_NT))
            gettimeofday(&t_tmp, NULL);
            time_gmres_pc = (double)t_tmp.tv_sec + (double)t_tmp.tv_usec/1000000 -  time_gmres_pc;
#else
            time_gmres_pc = ((double)clock()) / CLOCKS_PER_SEC - time_gmres_pc;
#endif
            std::ofstream fo;
	        fo.open("pctime.dat", std::ios::app);
            fo<<time_gmres_pc<<std::endl;
	        fo.close();
#endif
            x = my_gmres_solver(A, b, ilu0);
        }
        break;

    case LinSysPC::TYPE::BLOCK_ILU:
        {
            viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<RealFlow>, viennacl::linalg::ilu0_tag > block_ilu0(A, viennacl::linalg::ilu0_tag(), viennaclData::block_ilu_size);
            x = my_gmres_solver(A, b, block_ilu0);
        }
        break;

    case LinSysPC::TYPE::Chow_Patel_ILU0:
        {
            viennacl::linalg::chow_patel_tag chow_patel_ilu_config;
            chow_patel_ilu_config.sweeps(viennaclData::chow_patel_ilu_sweeps);
            chow_patel_ilu_config.jacobi_iters(viennaclData::chow_patel_ilu_jacobi_iters);
            viennacl::linalg::chow_patel_ilu_precond< viennacl::compressed_matrix<RealFlow> > chow_patel_ilu(A, chow_patel_ilu_config);
            x = my_gmres_solver(A, b, chow_patel_ilu);
        }
        break;
    default:
        mflog::log.set_one_processor_out();
        mflog::log<<std::endl<<"Error implicitmat type when setting mat for petsc!"<<std::endl;
        mflow_exit(mflow_error_flag(0, CPP_LINE));
        break;
    }

    // copy from device to host:
    copy(x.begin(), x.end(), stl_x.begin());
}

}

#endif
#endif