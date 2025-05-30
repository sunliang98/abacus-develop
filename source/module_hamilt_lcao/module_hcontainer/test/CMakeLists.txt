
if(ENABLE_LCAO)
AddTest(
  TARGET hcontainer_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_hcontainer.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../module_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET hcontainer_complex_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_hcontainer_complex.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../module_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET hcontainer_cost_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_hcontainer_time.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../module_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET hcontainer_folding_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_func_folding.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../func_folding.cpp ../../../module_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET hcontainer_transfer_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_transfer.cpp ../func_transfer.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../transfer.cpp ../../../module_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

install(FILES parallel_hcontainer_tests.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
find_program(BASH bash)
add_test(NAME hontainer_para_test
      COMMAND ${BASH} parallel_hcontainer_tests.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET hcontainer_output_test
  LIBS parameter base ${math_libs} device
  SOURCES test_hcontainer_output.cpp
    tmp_mocks.cpp
    ../output_hcontainer.cpp
    ../base_matrix.cpp
    ../hcontainer.cpp
    ../atom_pair.cpp
    ../../../module_basis/module_ao/parallel_orbitals.cpp
    ../../../module_io/sparse_matrix.cpp
)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET hcontainer_readCSR_test
  LIBS parameter base ${math_libs} device cell_info 
  SOURCES test_hcontainer_readCSR.cpp
    ../base_matrix.cpp
    ../hcontainer.cpp
    ../atom_pair.cpp
    ../output_hcontainer.cpp
    ../../../module_basis/module_ao/parallel_orbitals.cpp
    ../../../module_io/sparse_matrix.cpp
    ../../../module_io/csr_reader.cpp
    ../../../module_io/file_reader.cpp
    ../../../module_io/output.cpp
)
endif() 