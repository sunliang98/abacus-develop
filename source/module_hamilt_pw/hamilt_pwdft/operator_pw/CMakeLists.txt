list(APPEND operator_ks_pw_srcs
    operator_pw.cpp
    ekinetic_pw.cpp
    veff_pw.cpp
    nonlocal_pw.cpp
    meta_pw.cpp
    velocity_pw.cpp
    onsite_proj_pw.cpp
)

# this library is included in hamilt_pwdft now
#add_library(
#    operator_ks_pw
#    OBJECT
#    ${operator_ks_pw_srcs}
#)

#if(ENABLE_COVERAGE)
#  add_coverage(operator_ks_pw)
#endif()
