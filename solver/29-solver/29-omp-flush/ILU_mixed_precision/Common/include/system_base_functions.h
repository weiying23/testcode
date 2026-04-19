//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   system_base_functions.h
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

#ifndef MFL_SYSTEM_BASE_FUNCTIONS_H
#define MFL_SYSTEM_BASE_FUNCTIONS_H

// standard head files
#include <string>
#include <vector>
#include <cassert>
#include <fstream>


namespace mflow
{

// An macro-variable for line
#define CPP_LINE __LINE__

// An function to return error flag
// The id of file is suggested as the format "TXXYY", where
// T = 1 now.
// XX: The number represents the first letter of the file name.
//     a(01) b(02) c(03) d(04) e(05) f(06) g(07) h(08) i(09)
//     j(10) k(11) l(12) m(13) n(14) o(15) p(16) q(17) r(18) 
//     s(19) t(20) u(21) v(22) w(23) x(24) y(25) z(26)
// YY: A number to further distinguish different files with the
//     same first letter.
#define mflow_error_flag(file_id, line)  ((file_id*100000)+line)

// Exit program and output an error flag
void mflow_exit(const int error_flag);

// Create folder in current working directory with name fname
void CreatFolder(const std::string &fname);

// Create folder in current working directory with name fname for MPI
void CreatFolder_OneProcessor(const std::string &fname, const int porc = 1);

/// \brief Return true if the file is readable
/// \note  This function can be also used to check file existence
bool CheckFileReadable(const std::string & filename);

} // namespace mflow

#endif // MFL_SYSTEM_BASE_FUNCTIONS_H
