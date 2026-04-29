# Compile kernel
add_custom_target(lib_impl)
function(add_impl_share_lib NAME)
    add_library(${NAME} SHARED ${ARGN})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/host/
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/device/
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling
        ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling/include
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
    target_link_libraries(${NAME} PRIVATE shmem)
    add_dependencies(lib_impl ${NAME})
endfunction()

add_subdirectory(impl)

set(TILING_SOURCES
    tiling/coc_tiling_lut.cpp
    tiling/reducescatter_tiling.cpp
    tiling/allreduce_tiling.cpp
    tiling/allgather_tiling.cpp
)
add_library(tiling_lib SHARED ${TILING_SOURCES})
target_include_directories(tiling_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples/utils
)
target_compile_options(tiling_lib PRIVATE -O3)

add_executable(dynamic_tiling main.cpp)

target_include_directories(dynamic_tiling PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/host/
    ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/device/
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling
    ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling/include
    ${PROJECT_SOURCE_DIR}/examples/templates/include
    ${PROJECT_SOURCE_DIR}/examples/utils
)
target_link_options(dynamic_tiling PRIVATE --cce-fatobj-link)
target_link_libraries(dynamic_tiling PRIVATE tiling_lib shmem ${SHARE_LIB_LINK})
target_compile_options(dynamic_tiling PRIVATE -O3)

add_dependencies(dynamic_tiling lib_impl tiling_lib)
