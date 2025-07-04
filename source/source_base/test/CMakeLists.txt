remove_definitions(-D__MPI)
install(DIRECTORY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
AddTest(
  TARGET MODULE_BASE_blas_connector
  LIBS parameter  ${math_libs} base device
  SOURCES blas_connector_test.cpp
)
AddTest(
  TARGET MODULE_BASE_atom_in
  LIBS parameter 
  SOURCES atom_in_test.cpp
)
AddTest(
  TARGET MODULE_BASE_timer
  LIBS parameter 
  SOURCES timer_test.cpp ../timer.cpp  ../global_variable.cpp
)
AddTest(
  TARGET MODULE_BASE_tool_quit
  LIBS parameter 
  SOURCES tool_quit_test.cpp ../tool_quit.cpp ../global_variable.cpp ../global_file.cpp ../global_function.cpp ../memory.cpp ../timer.cpp  
)
AddTest(
  TARGET MODULE_BASE_tool_check
  LIBS parameter 
  SOURCES tool_check_test.cpp ../tool_check.cpp ../tool_quit.cpp ../global_variable.cpp ../global_file.cpp ../global_function.cpp ../memory.cpp ../timer.cpp 
)
AddTest(
  TARGET MODULE_BASE_memory
  SOURCES memory_test.cpp ../memory.cpp ../parallel_reduce.cpp
)
ADDTest(
  TARGET MODULE_BASE_global_function
  LIBS parameter ${math_libs}
  SOURCES global_function_test.cpp ../global_function.cpp ../tool_quit.cpp ../global_variable.cpp ../global_file.cpp ../memory.cpp ../timer.cpp
)
AddTest(
  TARGET MODULE_BASE_vector3
  LIBS parameter 
  SOURCES vector3_test.cpp
)
AddTest(
  TARGET MODULE_BASE_matrix3
  LIBS parameter  ${math_libs} base device
  SOURCES matrix3_test.cpp
)
AddTest(
  TARGET MODULE_BASE_intarray
  LIBS parameter 
  SOURCES intarray_test.cpp ../intarray.cpp
)
AddTest(
  TARGET MODULE_BASE_realarray
  LIBS parameter 
  SOURCES realarray_test.cpp ../realarray.cpp
)
AddTest(
 TARGET MODULE_BASE_matrix
  LIBS parameter  ${math_libs} base device
  SOURCES matrix_test.cpp
)
AddTest(
  TARGET MODULE_BASE_complexarray
  LIBS parameter 
  SOURCES complexarray_test.cpp ../complexarray.cpp ../tool_quit.cpp ../global_variable.cpp ../global_file.cpp ../global_function.cpp ../memory.cpp ../timer.cpp
)
AddTest(
  TARGET MODULE_BASE_complexmatrix
  LIBS parameter  ${math_libs} base device
  SOURCES complexmatrix_test.cpp
)
AddTest(
  TARGET MODULE_BASE_integral
  LIBS parameter 
  SOURCES math_integral_test.cpp ../math_integral.cpp
)
AddTest(
  TARGET MODULE_BASE_sph_bessel_recursive
  LIBS parameter 
  SOURCES sph_bessel_recursive_test.cpp ../sph_bessel_recursive-d1.cpp ../sph_bessel_recursive-d2.cpp
)
AddTest(
  TARGET MODULE_BASE_ylmreal
  LIBS parameter  ${math_libs} base device
  SOURCES math_ylmreal_test.cpp ../libm/branred.cpp ../libm/sincos.cpp
)
AddTest(
  TARGET MODULE_BASE_math_sphbes
  LIBS parameter 
  SOURCES math_sphbes_test.cpp ../math_sphbes.cpp ../timer.cpp
)
AddTest(
  TARGET MODULE_BASE_mathzone
  LIBS parameter  ${math_libs} base device
  SOURCES mathzone_test.cpp
)
AddTest(
  TARGET MODULE_BASE_mathzone_add1
  LIBS parameter  ${math_libs} base device
  SOURCES mathzone_add1_test.cpp
)
AddTest(
  TARGET MODULE_BASE_math_polyint
  LIBS parameter 
  SOURCES math_polyint_test.cpp ../math_polyint.cpp ../realarray.cpp ../timer.cpp
)
AddTest(
  TARGET MODULE_BASE_gram_schmidt_orth
  LIBS parameter  ${math_libs} base device
  SOURCES gram_schmidt_orth_test.cpp
)
AddTest(
  TARGET MODULE_BASE_math_bspline
  LIBS parameter 
  SOURCES math_bspline_test.cpp ../math_bspline.cpp
)
AddTest(
  TARGET MODULE_BASE_inverse_matrix
  LIBS parameter  ${math_libs} base device
  SOURCES inverse_matrix_test.cpp 
)
AddTest(
  TARGET MODULE_BASE_mymath
  LIBS parameter 
  SOURCES mymath_test.cpp ../mymath.cpp ../timer.cpp
)
AddTest(
  TARGET MODULE_BASE_container
  LIBS parameter 
  SOURCES container_operator_test.cpp ../container_operator.h
)

AddTest(
  TARGET MODULE_BASE_math_chebyshev
  LIBS parameter  ${math_libs} base device container
  SOURCES math_chebyshev_test.cpp
)

AddTest(
  TARGET MODULE_BASE_lapack_connector
  LIBS parameter  ${math_libs} base device
  SOURCES lapack_connector_test.cpp
)

AddTest(
  TARGET MODULE_BASE_opt_CG
  LIBS parameter  ${math_libs} base device
  SOURCES opt_CG_test.cpp opt_test_tools.cpp
)

AddTest(
  TARGET MODULE_BASE_opt_TN
  LIBS parameter  ${math_libs} base device
  SOURCES opt_TN_test.cpp opt_test_tools.cpp
)

AddTest(
  TARGET MODULE_BASE_ylm
  LIBS parameter 
  SOURCES ylm_test.cpp  ../ylm.cpp  ../timer.cpp ../tool_quit.cpp ../global_variable.cpp ../global_file.cpp ../global_function.cpp ../memory.cpp
)

AddTest(
  TARGET MODULE_BASE_abfs-vector3_order
  LIBS parameter 
  SOURCES abfs-vector3_order_test.cpp ../abfs-vector3_order.cpp
)

AddTest(
  TARGET MODULE_BASE_global_file
  LIBS parameter 
  SOURCES global_file_test.cpp ../global_file.cpp ../global_function.cpp ../tool_quit.cpp ../global_variable.cpp ../memory.cpp ../timer.cpp
)

AddTest(
  TARGET MODULE_BASE_tool_title
  LIBS parameter 
  SOURCES tool_title_test.cpp ../tool_title.cpp  ../global_variable.cpp ../global_function.cpp ../timer.cpp  ../tool_quit.cpp  ../global_file.cpp ../memory.cpp
)

AddTest(
  TARGET MODULE_BASE_element_basis_index
  LIBS parameter 
  SOURCES element_basis_index_test.cpp ../element_basis_index.cpp
)

AddTest(
  TARGET MODULE_BASE_tool_threading
  LIBS parameter 
  SOURCES tool_threading_test.cpp ../tool_threading.h
)

AddTest(
  TARGET MODULE_BASE_spherical_bessel_transformer
  SOURCES spherical_bessel_transformer_test.cpp
  LIBS parameter  ${math_libs} base device
)

AddTest(
  TARGET MODULE_BASE_cubic_spline
  SOURCES cubic_spline_test.cpp
  LIBS parameter  ${math_libs} base device
)

AddTest(
  TARGET MODULE_BASE_clebsch_gordan_coeff_test
  SOURCES clebsch_gordan_coeff_test.cpp
  LIBS parameter  ${math_libs} base device
)

AddTest(
  TARGET MODULE_BASE_assoc_laguerre_test
  SOURCES assoc_laguerre_test.cpp
  LIBS parameter  ${math_libs} base device
)

AddTest(
  TARGET MODULE_BASE_ndarray_test
  LIBS parameter 
  SOURCES ndarray_test.cpp
)

AddTest(
  TARGET MODULE_BASE_formatter_test
  LIBS parameter 
  SOURCES formatter_test.cpp
)

AddTest(
  TARGET MODULE_BASE_lebedev_laikov
  SOURCES test_lebedev_laikov.cpp ../ylm.cpp ../math_lebedev_laikov.cpp
)

if(ENABLE_GOOGLEBENCH)
  AddTest(
    TARGET MODULE_BASE_perf_sphbes
  LIBS parameter 
    SOURCES perf_sphbes_test.cpp ../math_sphbes.cpp ../timer.cpp
  )
endif()
