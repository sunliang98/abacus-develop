list(APPEND hamilt_ofdft_srcs
    kedf_tf.cpp
    kedf_vw.cpp
    kedf_wt.cpp
    kedf_lkt.cpp
    of_stress_pw.cpp
    kedf_ml.cpp
    kedf_ml_pot.cpp
    kedf_ml_label.cpp
    ml_data.cpp
    ml_data_position.cpp
    ml_tools/nn_of.cpp
)

add_library(
    hamilt_ofdft
    OBJECT
    ${hamilt_ofdft_srcs}
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_ofdft)
endif()
