#include "source_pw/hamilt_pwdft/kernels/meta_op.h"

#include <complex>

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <base/macros/macros.h>

namespace hamilt {

#define THREADS_PER_BLOCK 256


template <typename FPTYPE>
__global__ void meta_pw(
        const int ik,
        const int pol,
        const int npw,
        const int npwx,
        const FPTYPE tpiba,
        const FPTYPE* gcar,
        const FPTYPE* kvec_c,
        const thrust::complex<FPTYPE>* in,
        thrust::complex<FPTYPE>* out,
        const bool add)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw) {return;}

    FPTYPE fact = (gcar[(ik * npwx + ig) * 3 + pol] +
                   kvec_c[ik * 3 + pol]) * tpiba;
    if (add) {
        out[ig] -= in[ig] * thrust::complex<FPTYPE>(0.0, fact);
    }
    else {
        out[ig] = in[ig] * thrust::complex<FPTYPE>(0.0, fact);
    }
}

template <typename FPTYPE>
void meta_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                             const int& ik,
                                                             const int& pol,
                                                             const int& npw,
                                                             const int& npwx,
                                                             const FPTYPE& tpiba,
                                                             const FPTYPE* gcar,
                                                             const FPTYPE* kvec_c,
                                                             const std::complex<FPTYPE>* in,
                                                             std::complex<FPTYPE>* out,
                                                             const bool add)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    meta_pw<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        ik, pol, npw, npwx,
        tpiba,
        gcar, kvec_c,
        reinterpret_cast<const thrust::complex<FPTYPE> * >(in),
        reinterpret_cast<thrust::complex<FPTYPE> * >(out),
        add);

    cudaCheckOnDebug();
}

template struct meta_pw_op<float, base_device::DEVICE_GPU>;
template struct meta_pw_op<double, base_device::DEVICE_GPU>;

}  // namespace hamilt