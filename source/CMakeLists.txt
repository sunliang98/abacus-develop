add_subdirectory(module_base)
add_subdirectory(module_cell)
add_subdirectory(module_psi)
add_subdirectory(module_elecstate)
add_subdirectory(module_hamilt_general)
add_subdirectory(module_hamilt_pw)
add_subdirectory(module_hamilt_lcao)
add_subdirectory(module_hsolver)
add_subdirectory(module_basis/module_ao)
add_subdirectory(module_basis/module_nao)
add_subdirectory(module_md)
add_subdirectory(module_basis/module_pw)
add_subdirectory(module_esolver)
add_subdirectory(module_hamilt_lcao/module_gint)
add_subdirectory(module_io)
add_subdirectory(module_relax)
add_subdirectory(module_ri)
add_subdirectory(module_parameter)
add_subdirectory(module_lr)

# add by jghan
add_subdirectory(module_rdmft)

add_library(
    driver
    OBJECT
    driver.cpp
    driver_run.cpp
)

list(APPEND device_srcs
  module_hamilt_pw/hamilt_pwdft/kernels/nonlocal_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/veff_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/ekinetic_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/meta_op.cpp
  module_hamilt_pw/hamilt_stodft/kernels/hpsi_norm_op.cpp
  module_basis/module_pw/kernels/pw_op.cpp
  module_hsolver/kernels/dngvd_op.cpp
  module_hsolver/kernels/bpcg_kernel_op.cpp
  module_elecstate/kernels/elecstate_op.cpp

  # module_psi/kernels/psi_memory_op.cpp
  # module_psi/kernels/device.cpp

  module_base/module_device/device.cpp
  module_base/module_device/memory_op.cpp
  module_base/kernels/math_kernel_op.cpp
  module_base/kernels/math_kernel_op_vec.cpp

  module_hamilt_pw/hamilt_pwdft/kernels/force_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/stress_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/onsite_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/wf_op.cpp
  module_hamilt_pw/hamilt_pwdft/kernels/vnl_op.cpp
  module_base/kernels/math_ylm_op.cpp
  module_hamilt_general/module_xc/kernels/xc_functional_op.cpp
)

if(USE_CUDA)
  list(APPEND device_srcs
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/nonlocal_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/veff_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/ekinetic_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/meta_op.cu
    module_hamilt_pw/hamilt_stodft/kernels/cuda/hpsi_norm_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/onsite_op.cu
    module_basis/module_pw/kernels/cuda/pw_op.cu
    module_hsolver/kernels/cuda/dngvd_op.cu
    module_hsolver/kernels/cuda/bpcg_kernel_op.cu
    module_elecstate/kernels/cuda/elecstate_op.cu

    # module_psi/kernels/cuda/memory_op.cu
    module_base/module_device/cuda/memory_op.cu

    module_hamilt_pw/hamilt_pwdft/kernels/cuda/force_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/stress_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/wf_op.cu
    module_hamilt_pw/hamilt_pwdft/kernels/cuda/vnl_op.cu
    module_base/kernels/cuda/math_ylm_op.cu
    module_base/kernels/cuda/math_kernel_op.cu
    module_base/kernels/cuda/math_kernel_op_vec.cu
    module_hamilt_general/module_xc/kernels/cuda/xc_functional_op.cu
  )
endif()

if(USE_ROCM)
  hip_add_library(device_rocm STATIC
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/nonlocal_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/veff_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/ekinetic_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/meta_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/onsite_op.hip.cu
    module_hamilt_pw/hamilt_stodft/kernels/rocm/hpsi_norm_op.hip.cu
    module_basis/module_pw/kernels/rocm/pw_op.hip.cu
    module_hsolver/kernels/rocm/dngvd_op.hip.cu
    module_hsolver/kernels/rocm/bpcg_kernel_op.hip.cu
    module_elecstate/kernels/rocm/elecstate_op.hip.cu

    # module_psi/kernels/rocm/memory_op.hip.cu
    module_base/module_device/rocm/memory_op.hip.cu

    module_hamilt_pw/hamilt_pwdft/kernels/rocm/force_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/stress_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/wf_op.hip.cu
    module_hamilt_pw/hamilt_pwdft/kernels/rocm/vnl_op.hip.cu
    module_base/kernels/rocm/math_kernel_op.hip.cu
    module_base/kernels/rocm/math_kernel_op_vec.hip.cu
    module_base/kernels/rocm/math_ylm_op.hip.cu
    module_hamilt_general/module_xc/kernels/rocm/xc_functional_op.hip.cu
  )
endif()

if(USE_DSP)
  list(APPEND device_srcs
  module_base/kernels/dsp/dsp_connector.cpp
  )
endif()


add_library(device OBJECT ${device_srcs})

if(USE_CUDA)
  target_link_libraries(
    device 
    cusolver 
    cublas 
    cufft
  )
elseif(USE_ROCM)
  target_link_libraries(
    device 
    device_rocm 
    hip::host
    hip::device
    hip::hipfft
    roc::hipblas
    roc::hipsolver
  )
endif()

if(ENABLE_COVERAGE)
  add_coverage(driver)
endif()
