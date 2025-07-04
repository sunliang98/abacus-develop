remove_definitions(-D __MPI)

add_library(tddft_test_lib tddft_test.cpp)
target_link_libraries(tddft_test_lib Threads::Threads GTest::gtest_main GTest::gmock_main)
#target_include_directories(tddft_test_lib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${GTEST_INCLUDE_DIRS}>)

AddTest(
  TARGET tddft_middle_hamilt_test
  LIBS parameter ${math_libs} base device tddft_test_lib  
  SOURCES middle_hamilt_test.cpp ../middle_hamilt.cpp 
)

AddTest(
  TARGET tddft_band_energy_test
  LIBS parameter ${math_libs} base device tddft_test_lib  
  SOURCES band_energy_test.cpp ../band_energy.cpp ../../../source_basis/module_ao/parallel_orbitals.cpp
)

AddTest(
  TARGET tddft_norm_psi_test
  LIBS parameter ${math_libs} base device tddft_test_lib  
  SOURCES norm_psi_test.cpp ../norm_psi.cpp 
)

AddTest(
  TARGET tddft_upsi_test
  LIBS parameter ${math_libs} base device tddft_test_lib  
  SOURCES upsi_test1.cpp upsi_test2.cpp upsi_test3.cpp ../upsi.cpp 
)

AddTest(
  TARGET tddft_propagator_test
  LIBS parameter ${math_libs} base device tddft_test_lib  
  SOURCES propagator_test1.cpp propagator_test2.cpp propagator_test3.cpp ../propagator.cpp ../propagator_cn2.cpp ../propagator_taylor.cpp ../propagator_etrs.cpp
)

