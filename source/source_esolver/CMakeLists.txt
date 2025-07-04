list(APPEND objects
    esolver.cpp
    esolver_ks.cpp
    esolver_fp.cpp
    esolver_ks_pw.cpp
    esolver_ks_lcaopw.cpp
    esolver_sdft_pw.cpp
    esolver_lj.cpp
    esolver_dp.cpp
    esolver_of.cpp
    esolver_of_interface.cpp
    esolver_of_tool.cpp
    pw_others.cpp
    pw_setup.cpp
)
if(ENABLE_LCAO)
  list(APPEND objects
      esolver_ks_lcao.cpp
      esolver_ks_lcao_tddft.cpp
      lcao_before_scf.cpp
      lcao_after_scf.cpp
      esolver_gets.cpp
      lcao_others.cpp
      esolver_dm2rho.cpp
  )
endif()

add_library(
    esolver
    OBJECT
    ${objects}
        ../source_pw/hamilt_pwdft/module_exx_helper/exx_helper.cpp
        ../source_pw/hamilt_pwdft/module_exx_helper/exx_helper.h
)

if(ENABLE_COVERAGE)
  add_coverage(esolver)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()

