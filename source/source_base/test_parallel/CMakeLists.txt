AddTest(
  TARGET MODULE_BASE_ParaCommon
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_common_test.cpp ../global_variable.cpp ../parallel_common.cpp
)

AddTest(
  TARGET MODULE_BASE_ParaGlobal
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_global_test.cpp ../global_variable.cpp ../parallel_global.cpp ../parallel_comm.cpp
)

AddTest(
  TARGET MODULE_BASE_ParaReduce
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_reduce_test.cpp ../global_variable.cpp ../parallel_global.cpp ../parallel_comm.cpp ../parallel_common.cpp ../parallel_reduce.cpp
)

install(FILES parallel_common_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES parallel_global_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES parallel_reduce_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_program(BASH bash)
add_test(NAME MODULE_BASE_parallel_common_test
      COMMAND ${BASH} parallel_common_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_test(NAME MODULE_BASE_parallel_global_test
      COMMAND ${BASH} parallel_global_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_test(NAME MODULE_BASE_parallel_reduce_test
      COMMAND ${BASH} parallel_reduce_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET MODULE_BASE_para_gemm
  LIBS MPI::MPI_CXX ${math_libs} base device parameter
  SOURCES test_para_gemm.cpp
)

AddTest(
  TARGET MODULE_BASE_math_chebyshev_mpi
  LIBS MPI::MPI_CXX parameter ${math_libs} base device container
  SOURCES math_chebyshev_mpi_test.cpp
)

add_test(NAME MODULE_BASE_para_gemm_parallel
      COMMAND mpirun -np 4 ./MODULE_BASE_para_gemm
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
    TARGET MODULE_BASE_parallel_2d_test
    SOURCES parallel_2d_test.cpp ../parallel_2d.cpp
    LIBS parameter ${math_libs} 
)

install(FILES parallel_2d_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
find_program(BASH bash)
add_test(NAME MODULE_BASE_parallel_2d_test_para
      COMMAND ${BASH} parallel_2d_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


  # figure out the lib that provides BLACS
  if(MKLROOT)
    list(APPEND BLACS_LIB MKL::MKL MKL::MKL_SCALAPACK)
  else()
    set(BLACS_LIB ScaLAPACK::ScaLAPACK)
  endif()

  AddTest(
    TARGET MODULE_BASE_blacs_connector
    LIBS parameter MPI::MPI_CXX ${BLACS_LIB}
    SOURCES blacs_connector_test.cpp
  )
  install(FILES blacs_connector_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME MODULE_BASE_blacs_connector_test
        COMMAND ${BASH} blacs_connector_test.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

