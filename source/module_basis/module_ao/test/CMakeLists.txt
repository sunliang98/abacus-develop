remove_definitions(-D__EXX)

list(APPEND depend_files
 ../../../module_base/math_integral.cpp
  ../../../module_base/math_sphbes.cpp
  ../../../module_base/math_polyint.cpp
  ../../../module_base/math_ylmreal.cpp
  ../../../module_base/ylm.cpp
  ../../../module_base/memory.cpp
  ../../../module_base/blas_connector.cpp
  ../../../module_base/complexarray.cpp
  ../../../module_base/complexmatrix.cpp
  ../../../module_base/matrix.cpp
  ../../../module_base/realarray.cpp
  ../../../module_base/intarray.cpp
  ../../../module_base/sph_bessel_recursive-d1.cpp
  ../../../module_base/sph_bessel_recursive-d2.cpp
  ../../../module_base/tool_title.cpp
  ../../../module_base/tool_quit.cpp
  ../../../module_base/tool_check.cpp
  ../../../module_base/timer.cpp
  ../../../module_base/mathzone_add1.cpp
  ../../../module_base/global_variable.cpp
  ../../../module_base/global_function.cpp
  ../../../module_base/global_file.cpp
  ../../../module_base/libm/branred.cpp
  ../../../module_base/libm/sincos.cpp
  ../../../module_base/spherical_bessel_transformer.cpp
  ../../../module_base/cubic_spline.cpp
  ../../../module_base/parallel_2d.cpp
  ../ORB_read.cpp
  ../ORB_atomic.cpp
  ../ORB_atomic_lm.cpp
  ../ORB_nonlocal.cpp
  ../ORB_nonlocal_lm.cpp
  ../ORB_gaunt_table.cpp
  ../parallel_orbitals.cpp
  ../../../module_hamilt_lcao/hamilt_lcaodft/center2_orb-orb11.cpp
  ../../../module_hamilt_lcao/hamilt_lcaodft/center2_orb.cpp
  )

install(DIRECTORY GaAs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../../../tests)
install(DIRECTORY GaAs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


AddTest(
    TARGET ORB_nonlocal_test
    SOURCES ORB_nonlocal_test.cpp
        ../ORB_nonlocal.cpp
        ../ORB_nonlocal_lm.cpp
    LIBS parameter ${math_libs} device base
)

AddTest(
    TARGET ORB_nonlocal_lm_test
    SOURCES ORB_nonlocal_lm_test.cpp
        ../ORB_nonlocal_lm.cpp
    LIBS parameter ${math_libs} device base
)

AddTest(
    TARGET ORB_atomic_lm_test
    SOURCES ORB_atomic_lm_test.cpp
        ../ORB_atomic_lm.cpp
    LIBS parameter ${math_libs} device base
)

AddTest(
    TARGET ORB_read_test
    SOURCES ORB_read_test.cpp
        ../ORB_read.cpp
        ../ORB_atomic.cpp
        ../ORB_atomic_lm.cpp
    LIBS parameter ${math_libs} device base 
)


AddTest(
    TARGET parallel_orbitals_test 
    SOURCES parallel_orbitals_test.cpp ../parallel_orbitals.cpp
    LIBS parameter ${math_libs} device base
)

install(FILES parallel_orbitals_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
find_program(BASH bash)
add_test(NAME parallel_orbitals_test_para
      COMMAND ${BASH} parallel_orbitals_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(DIRECTORY lcao_H2O DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY lcao_H2O DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../../../tests)

