remove_definitions(-D__MPI)
remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)

AddTest(
    TARGET Elecstate_Kernels_UTs
    LIBS parameter ${math_libs} psi base device
    SOURCES elecstate_op_test.cpp
)
