list(APPEND hamilt_ofdft_srcs
    kedf_tf.cpp
    kedf_vw.cpp
    kedf_wt.cpp
    kedf_lkt.cpp
    kedf_manager.cpp
    of_stress_pw.cpp
)

add_library(
    hamilt_ofdft
    OBJECT
    ${hamilt_ofdft_srcs}
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_ofdft)
endif()

if(ENABLE_MLALGO)
  list(APPEND hamilt_mlkedf_srcs
      kedf_ml.cpp
      kedf_ml_pot.cpp
      kedf_ml_label.cpp
      ml_tools/nn_of.cpp
  )

  add_library(
      hamilt_mlkedf
      OBJECT
      ${hamilt_mlkedf_srcs}
  )

  if(ENABLE_COVERAGE)
    add_coverage(hamilt_mlkedf)
  endif()
endif()