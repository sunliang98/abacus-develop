remove_definitions(-D__MLALGO)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(TARGET dm_io_test_serial)
  remove_definitions(-D__MPI)
endif()

AddTest(
  TARGET dm_io_test_serial
  LIBS parameter ${math_libs} base device cell_info 
  SOURCES test_dm_io.cpp ../density_matrix.cpp ../density_matrix_io.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
  ${ABACUS_SOURCE_DIR}/source_io/output.cpp
)

AddTest(
  TARGET dm_constructor_test
  LIBS parameter  ${math_libs} base device
  SOURCES test_dm_constructor.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)

AddTest(
  TARGET dm_init_test
  LIBS parameter  ${math_libs} base device
  SOURCES test_dm_R_init.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)

AddTest(
  TARGET dm_cal_DMR_test
  LIBS parameter  ${math_libs} base device
  SOURCES test_cal_dm_R.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)
