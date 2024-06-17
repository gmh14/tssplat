cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc)

project(tet_spheres LANGUAGES C CXX CUDA VERSION 1.0)

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(pgo CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")


# tweak torch dependencies, see https://github.com/pytorch/pytorch/issues/33928
get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)

message(STATUS "torch int lib:${TORCH_INTERFACE_LIB}")
message(STATUS "torch include:${TORCH_INCLUDE_DIRS}")
message(STATUS "torch lib:${TORCH_LIBRARIES}")

# string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
# set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES ${TORCH_INTERFACE_LIB})

# add library
add_library(tet_spheres_ext MODULE
    tet_spheres/tet_spheres.cpp
    tet_spheres/tet_spheres_cuda.cu
)

python_extension_module(tet_spheres_ext)

target_include_directories(tet_spheres_ext PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(tet_spheres_ext ${TORCH_LIBRARIES})
target_link_libraries(tet_spheres_ext ${CUDA_cusparse_LIBRARY})
target_link_libraries(tet_spheres_ext ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(tet_spheres_ext ${CUDA_LIBRARIES})
target_link_libraries(tet_spheres_ext ${TORCH_PYTHON_LIBRARY})
target_link_libraries(tet_spheres_ext pgo::pgo_c)

set_property(TARGET tet_spheres_ext PROPERTY CXX_STANDARD 17)

install(TARGETS tet_spheres_ext DESTINATION tet_spheres)
