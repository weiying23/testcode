#==============================================================================
#         MFlow CMakeLists.txt files for CMake tools
# This file deals with common h and cpp files of MFlow
# All messages by author begin with symbols "%%", distinguished with system info 
# Author: tangj 2019-03-01
#==============================================================================

# CMake version required
cmake_minimum_required(VERSION 3.9)

message(STATUS "%%-------------------------------------------------------------")
message(STATUS "%% Enter directory Common to deal with common h and cpp files")
message(STATUS "%%-------------------------------------------------------------")

#qleonardo: link DC-lib
find_library(DC_lib libuTaskTree_OMP.so ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
if (NOT DC_lib)
	message(FATAL_ERROR "DC_lib library cannot be found!")
endif()
link_libraries(${DC_lib})

# set library name for common files and 
# ***Export the name of Common lib upto the global scope*** 
set(LINK_COMM_LIB Common CACHE INTERNAL "name of Common lib")
set(LINK_PCOMM_LIB pCommon CACHE INTERNAL "name of pCommon lib")

# ***Export the include directory of Common lib upto the global scope***
set(COMM_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "inlcude directory of Common lib")

set(COMM_CUINC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuinclude)

# introduce the common include directories    
include_directories(AFTER ${EXTRA_LIBS_INC_DIRS} ${COMM_INC_DIR} ${COMM_CUINC_DIR} ${MPI_INC} ${MPI_INCLUDE_PATH})

# Set source file directory
set(COMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(COMM_CUSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cusrc)


# source files lists
file(GLOB INCLIST_COMMON  "${COMM_INC_DIR}/*.h" "${COMM_CUINC_DIR}/*.cuh")
if(enable_GPU)
file(GLOB SRCLIST_COMMON  "${COMM_CUSRC_DIR}/*.cu" "${COMM_SRC_DIR}/*.cpp")
else(enable_GPU)
file(GLOB SRCLIST_COMMON  ${COMM_SRC_DIR}/*.cpp)
endif()

# group source files for visual studio
source_group("${LIB_COMM_NAME}\\include"  FILES  ${INCLIST_COMMON})
source_group("${LIB_COMM_NAME}\\source"   FILES  ${SRCLIST_COMMON})

# add definitions:
if(BUILD_SHARED_LIBRARY)
    message(STATUS "%% Add definitions -DEXPORTDLL")
    add_definitions(-DEXPORTDLL)
endif(BUILD_SHARED_LIBRARY)

# target output directory
set(LIBRARY_OUTPUT_PATH  ${CMAKE_CURRENT_BINARY_DIR}/lib/${DIR_OS_ARC})

# ***Export the directory of Common lib upto the global scope*** 
set(LINK_COMMON_DIR ${LIBRARY_OUTPUT_PATH} CACHE INTERNAL "linking directory of Common")

# add library targets
#IF(enable_Parallel_MPI)
#    add_definitions(-DMPICH -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX) 
#ENDIF(enable_Parallel_MPI)


#IF(enable_Parallel_MPI)
#    remove_definitions(-DMPICH -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX) 
#ENDIF(enable_Parallel_MPI)

if(enable_Parallel_MPI)
	# add parallel lib of Common
	ADD_LIBRARY(${LINK_PCOMM_LIB} ${LIBRARY_TYPE}  ${INCLIST_COMMON}  ${SRCLIST_COMMON})
	if(WIN32)
	  set_target_properties(${LINK_PCOMM_LIB} PROPERTIES COMPILE_FLAGS "-DMPICH -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -DMFMESH_HAVE_MPI /MP")
	else()
	  set_target_properties(${LINK_PCOMM_LIB} PROPERTIES COMPILE_FLAGS "-DMPICH -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -DMFMESH_HAVE_MPI")
	endif(WIN32)
	message(STATUS "%% Add library " ${LINK_PCOMM_LIB})
else(enable_Parallel_MPI)
	# add serial lib of Common
	ADD_LIBRARY(${LINK_COMM_LIB} ${LIBRARY_TYPE}  ${INCLIST_COMMON}  ${SRCLIST_COMMON})
	if(WIN32)
	  set_target_properties(${LINK_COMM_LIB} PROPERTIES COMPILE_FLAGS "/MP")
	endif(WIN32)
	message(STATUS "%% Add library " ${LINK_COMM_LIB})
endif(enable_Parallel_MPI)



