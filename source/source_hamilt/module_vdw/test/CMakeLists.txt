remove_definitions(-D__MLALGO)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)

install(FILES c6.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES r0.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET vdwTest
  LIBS parameter ${math_libs} base device vdw 
  SOURCES vdw_test.cpp 
)
