list(APPEND objects
    spin_constrain.cpp
    init_sc.cpp
    cal_mw.cpp
    basic_funcs.cpp
    lambda_loop_helper.cpp
    lambda_loop.cpp
    cal_mw_from_lambda.cpp
    template_helpers.cpp
)

add_library(
    deltaspin
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(deltaspin)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()

