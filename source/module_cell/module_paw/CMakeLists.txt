add_library(
    paw
    OBJECT
    paw_element.cpp
    paw_sphbes.cpp
    paw_cell.cpp
    paw_cell_libpaw.cpp
    paw_atom.cpp
)

target_link_libraries(paw libpaw_interface ${math_libs})
add_dependencies(paw libpaw_interface)

if(ENABLE_COVERAGE)
  add_coverage(paw)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()