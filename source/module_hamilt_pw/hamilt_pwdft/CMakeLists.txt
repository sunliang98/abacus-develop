add_subdirectory(operator_pw)

list(APPEND objects
    hamilt_pw.cpp
    operator_pw/ekinetic_pw.cpp
    operator_pw/veff_pw.cpp
    operator_pw/nonlocal_pw.cpp
    operator_pw/meta_pw.cpp
    operator_pw/velocity_pw.cpp
    operator_pw/operator_pw.cpp
    operator_pw/onsite_proj_pw.cpp
    operator_pw/op_exx_pw.cpp
    forces_nl.cpp
    forces_cc.cpp
    forces_scc.cpp
    forces.cpp
    forces_us.cpp
    forces_onsite.cpp
    stress_func_cc.cpp
    stress_func_ewa.cpp
    stress_func_gga.cpp
    stress_func_mgga.cpp
    stress_func_har.cpp
    stress_func_kin.cpp
    stress_func_loc.cpp
    stress_func_nl.cpp
    stress_func_us.cpp
    stress_func_onsite.cpp
    stress_pw.cpp
    VL_in_pw.cpp
    VNL_in_pw.cpp
    VNL_grad_pw.cpp
    structure_factor.cpp
    structure_factor_k.cpp
    soc.cpp
    global.cpp
    parallel_grid.cpp
    elecond.cpp
    fs_nonlocal_tools.cpp
    fs_kin_tools.cpp
    radial_proj.cpp
    onsite_projector.cpp
    onsite_proj_tools.cpp
)

add_library(
    hamilt_pwdft
    OBJECT
    ${objects}
        stress_func_exx.cpp
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_pwdft)
endif()

if (BUILD_TESTING)
    if(ENABLE_MPI)
        add_subdirectory(test)
        add_subdirectory(kernels/test)
    endif()
endif()
