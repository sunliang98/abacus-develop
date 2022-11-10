cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nnof)

list(APPEND CMAKE_PREFIX_PATH /home/dell/2_software/LibTorch-1.11.0/libtorch-cpu-pre-1.11.0/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)
add_executable(nnof train.cpp nn_of.cpp train_fft.cpp train_input.cpp train_data.cpp)
target_link_libraries(nnof "${TORCH_LIBRARIES}")
set_property(TARGET nnof PROPERTY CXX_STANDARD 14)