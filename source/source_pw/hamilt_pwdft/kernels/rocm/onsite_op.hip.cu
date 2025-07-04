#include "source_pw/hamilt_pwdft/kernels/onsite_op.h"

#include <base/macros/macros.h>
#include <complex>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

namespace hamilt
{

#define THREADS_PER_BLOCK 256

template <typename FPTYPE>
__global__ void onsite_op(const int npm,
                          const int npol,
                          const int* ip_iat,
                          const int tnp,
                          const thrust::complex<FPTYPE>* lambda_coeff,
                          thrust::complex<FPTYPE>* ps,
                          const thrust::complex<FPTYPE>* becp)
{
    const int ip = blockIdx.x;
    const int nbands = npm / npol;
    for (int ib = threadIdx.x; ib < nbands; ib += blockDim.x)
    {
        int ib2 = ib * npol;
        int iat = ip_iat[ip];
        const int psind = ip * npm + ib2;
        const int becpind = ib2 * tnp + ip;
        ps[psind] += lambda_coeff[iat * 4] * becp[becpind] + lambda_coeff[iat * 4 + 2] * becp[becpind + tnp];
        ps[psind + 1] += lambda_coeff[iat * 4 + 1] * becp[becpind] + lambda_coeff[iat * 4 + 3] * becp[becpind + tnp];
    }
}

template <typename FPTYPE>
__global__ void onsite_op(const int npm,
                          const int npol,
                          const int* orb_l_iat,
                          const int* ip_iat,
                          const int* ip_m,
                          const int* vu_begin_iat,
                          const int tnp,
                          const thrust::complex<FPTYPE>* vu,
                          thrust::complex<FPTYPE>* ps,
                          const thrust::complex<FPTYPE>* becp)
{
    const int ip = blockIdx.x;
    int m1 = ip_m[ip];
    if (m1 >= 0)
    {
        const int nbands = npm / npol;
        for (int ib = threadIdx.x; ib < nbands; ib += blockDim.x)
        {
            int ib2 = ib * npol;
            int iat = ip_iat[ip];
            const thrust::complex<FPTYPE>* vu_iat = vu + vu_begin_iat[iat];
            int orb_l = orb_l_iat[iat];
            int tlp1 = 2 * orb_l + 1;
            int tlp1_2 = tlp1 * tlp1;
            int ip2_begin = ip - m1;
            int ip2_end = ip - m1 + tlp1;
            const int psind = ip * npm + ib2;
            for (int ip2 = ip2_begin; ip2 < ip2_end; ip2++)
            {
                const int becpind = ib2 * tnp + ip2;
                int m2 = ip_m[ip2];
                const int index_mm = m1 * tlp1 + m2;
                ps[psind] += vu_iat[index_mm] * becp[becpind] + vu_iat[index_mm + tlp1_2 * 2] * becp[becpind + tnp];
                ps[psind + 1] += vu_iat[index_mm + tlp1_2 * 1] * becp[becpind]
                                 + vu_iat[index_mm + tlp1_2 * 3] * becp[becpind + tnp];
            }
        }
    }
}

template <typename FPTYPE>
void hamilt::onsite_ps_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                                       const int& npm,
                                                                       const int npol,
                                                                       const int* ip_iat,
                                                                       const int& tnp,
                                                                       const std::complex<FPTYPE>* lambda_coeff,
                                                                       std::complex<FPTYPE>* ps,
                                                                       const std::complex<FPTYPE>* becp)
{
    // denghui implement 20221019
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    hipLaunchKernelGGL(HIP_KERNEL_NAME(onsite_op<FPTYPE>), dim3(tnp), dim3(THREADS_PER_BLOCK), 0, 0,
        npm,
        npol,
        ip_iat,
        tnp,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(lambda_coeff),
        reinterpret_cast<thrust::complex<FPTYPE>*>(ps),          // array of data
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp)); // array of data

    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::onsite_ps_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                                       const int& npm,
                                                                       const int npol,
                                                                       const int* orb_l_iat,
                                                                       const int* ip_iat,
                                                                       const int* ip_m,
                                                                       const int* vu_begin_iat,
                                                                       const int& tnp,
                                                                       const std::complex<FPTYPE>* vu,
                                                                       std::complex<FPTYPE>* ps,
                                                                       const std::complex<FPTYPE>* becp)
{
    // denghui implement 20221109
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    hipLaunchKernelGGL(HIP_KERNEL_NAME(onsite_op<FPTYPE>), dim3(tnp), dim3(THREADS_PER_BLOCK), 0, 0,
        npm,
        npol,
        orb_l_iat,
        ip_iat,
        ip_m,
        vu_begin_iat,
        tnp,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(vu),
        reinterpret_cast<thrust::complex<FPTYPE>*>(ps),          // array of data
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp)); // array of data

    hipCheckOnDebug();
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
}

template struct onsite_ps_op<float, base_device::DEVICE_GPU>;
template struct onsite_ps_op<double, base_device::DEVICE_GPU>;

} // namespace hamilt
