remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__EXX)
remove_definitions(-D__LCAO)
remove_definitions(-DUSE_PAW)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES unitcell_test_pw_para.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET cell_unitcell_test_pw
  LIBS parameter ${math_libs} base device 
  SOURCES unitcell_test_pw.cpp ../unitcell.cpp ../read_atoms.cpp ../atom_spec.cpp ../update_cell.cpp ../bcast_cell.cpp
	../atom_pseudo.cpp ../pseudo.cpp ../read_pp.cpp ../read_pp_complete.cpp ../read_pp_upf201.cpp ../read_pp_upf100.cpp
  ../read_stru.cpp   ../read_atom_species.cpp
	../read_pp_vwr.cpp ../read_pp_blps.cpp ../../module_io/output.cpp 
  ../../module_elecstate/read_pseudo.cpp ../../module_elecstate/cal_nelec_nband.cpp
  ../../module_elecstate/read_orb.cpp    ../../module_cell/print_cell.cpp
  ../../module_elecstate/cal_wfc.cpp
)

find_program(BASH bash)
add_test(NAME cell_unitcell_test_pw_para
      COMMAND ${BASH} unitcell_test_pw_para.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
