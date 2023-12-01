cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nnof)

list(APPEND CMAKE_PREFIX_PATH /home/xianyuer/data/1_sunliang/2_software/libtorch/Libtorch-1.11.0/libtorch/share/cmake/Torch/)
#list(APPEND CMAKE_PREFIX_PATH /home/xianyuer/data/1_sunliang/2_software/libtorch/Libtorch-2.1.0/libtorch/share/cmake/Torch/)
#list(APPEND CMAKE_PREFIX_PATH /home/dell/2_software/LibTorch-1.11.0/libtorch-cpu-pre-1.11.0/libtorch/share/cmake/Torch/)
#list(APPEND CMAKE_PREFIX_PATH /home/dell/2_software/LibTorch-1.11.0/libtorch-cu102-1.11.0/libtorch/share/cmake/Torch/)
#list(APPEND CMAKE_PREFIX_PATH /home/dell/2_software/LibTorch-1.11.0/libtorch-cpu-1.11.0/libtorch/share/cmake/Torch/)
#list(APPEND CMAKE_PREFIX_PATH /home/dell/2_software/LibTorch-1.13.0/libtorch/share/cmake/Torch/)

find_package(Torch REQUIRED)
add_executable(nnof main.cpp data.cpp nn_of.cpp grid.cpp input.cpp kernel.cpp potential.cpp train.cpp)
target_link_libraries(nnof "${TORCH_LIBRARIES}")
set_property(TARGET nnof PROPERTY CXX_STANDARD 14)
