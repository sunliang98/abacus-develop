#include "module_hsolver/kernels/dngvd_op.h"

#include <hip/hip_runtime.h>
#include <base/macros/macros.h>

namespace hsolver {

void createGpuSolverHandle() {
    return;
}

void destroyGpuSolverHandle() {
    return;
}

#ifdef __LCAO
template <>
void dngvd_op<double, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                           const int nstart,
                                                           const int ldh,
                                                           const double* _hcc,
                                                           const double* _scc,
                                                           double* _eigenvalue,
                                                           double* _vcc)
{
    std::vector<double> hcc(nstart * nstart, 0.0);
    std::vector<double> scc(nstart * nstart, 0.0);
    std::vector<double> vcc(nstart * nstart, 0.0);
    std::vector<double> eigenvalue(nstart, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(double) * hcc.size(), hipMemcpyDeviceToHost));
    hipErrcheck(hipMemcpy(scc.data(), _scc, sizeof(double) * scc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dngvd_op<double, base_device::DEVICE_CPU>()(cpu_ctx,
                                                nstart,
                                                ldh,
                                                hcc.data(),
                                                scc.data(),
                                                eigenvalue.data(),
                                                vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(double) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
}
#endif // __LCAO

template <>
void dngvd_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                        const int nstart,
                                                                        const int ldh,
                                                                        const std::complex<float>* _hcc,
                                                                        const std::complex<float>* _scc,
                                                                        float* _eigenvalue,
                                                                        std::complex<float>* _vcc)
{
    std::vector<std::complex<float>> hcc(nstart * nstart, {0, 0});
    std::vector<std::complex<float>> scc(nstart * nstart, {0, 0});
    std::vector<std::complex<float>> vcc(nstart * nstart, {0, 0});
    std::vector<float> eigenvalue(nstart, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<float>) * hcc.size(), hipMemcpyDeviceToHost));
    hipErrcheck(hipMemcpy(scc.data(), _scc, sizeof(std::complex<float>) * scc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dngvd_op<std::complex<float>, base_device::DEVICE_CPU>()(cpu_ctx,
                                                             nstart,
                                                             ldh,
                                                             hcc.data(),
                                                             scc.data(),
                                                             eigenvalue.data(),
                                                             vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<float>) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(float) * eigenvalue.size(), hipMemcpyHostToDevice));
}

template <>
void dngvd_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                         const int nstart,
                                                                         const int ldh,
                                                                         const std::complex<double>* _hcc,
                                                                         const std::complex<double>* _scc,
                                                                         double* _eigenvalue,
                                                                         std::complex<double>* _vcc)
{
    std::vector<std::complex<double>> hcc(nstart * nstart, {0, 0});
    std::vector<std::complex<double>> scc(nstart * nstart, {0, 0});
    std::vector<std::complex<double>> vcc(nstart * nstart, {0, 0});
    std::vector<double> eigenvalue(nstart, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<double>) * hcc.size(), hipMemcpyDeviceToHost));
    hipErrcheck(hipMemcpy(scc.data(), _scc, sizeof(std::complex<double>) * scc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dngvd_op<std::complex<double>, base_device::DEVICE_CPU>()(cpu_ctx,
                                                              nstart,
                                                              ldh,
                                                              hcc.data(),
                                                              scc.data(),
                                                              eigenvalue.data(),
                                                              vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<double>) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
}

#ifdef __LCAO
template <>
void dnevx_op<double, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                           const int nstart,
                                                           const int ldh,
                                                           const double* _hcc,
                                                           const int m,
                                                           double* _eigenvalue,
                                                           double* _vcc)
{
    std::vector<double> hcc(ldh * ldh, 0.0);
    std::vector<double> vcc(ldh * ldh, 0.0);
    std::vector<double> eigenvalue(ldh, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(double) * hcc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dnevx_op<double, base_device::DEVICE_CPU>()(cpu_ctx, nstart, ldh, hcc.data(), m, eigenvalue.data(), vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(double) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
}
#endif // __LCAO

template <>
void dnevx_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                        const int nstart,
                                                                        const int ldh,
                                                                        const std::complex<float>* _hcc,
                                                                        const int m,
                                                                        float* _eigenvalue,
                                                                        std::complex<float>* _vcc)
{
    std::vector<std::complex<float>> hcc(ldh * ldh, {0, 0});
    std::vector<std::complex<float>> vcc(ldh * ldh, {0, 0});
    std::vector<float> eigenvalue(ldh, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<float>) * hcc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dnevx_op<std::complex<float>, base_device::DEVICE_CPU>()(cpu_ctx,
                                                             nstart,
                                                             ldh,
                                                             hcc.data(),
                                                             m,
                                                             eigenvalue.data(),
                                                             vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<float>) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(float) * eigenvalue.size(), hipMemcpyHostToDevice));
}

template <>
void dnevx_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                         const int nstart,
                                                                         const int ldh,
                                                                         const std::complex<double>* _hcc,
                                                                         const int m,
                                                                         double* _eigenvalue,
                                                                         std::complex<double>* _vcc)
{
    std::vector<std::complex<double>> hcc(ldh * ldh, {0, 0});
    std::vector<std::complex<double>> vcc(ldh * ldh, {0, 0});
    std::vector<double> eigenvalue(ldh, 0);
    hipErrcheck(hipMemcpy(hcc.data(), _hcc, sizeof(std::complex<double>) * hcc.size(), hipMemcpyDeviceToHost));
    base_device::DEVICE_CPU* cpu_ctx = {};
    dnevx_op<std::complex<double>, base_device::DEVICE_CPU>()(cpu_ctx,
                                                              nstart,
                                                              ldh,
                                                              hcc.data(),
                                                              m,
                                                              eigenvalue.data(),
                                                              vcc.data());
    hipErrcheck(hipMemcpy(_vcc, vcc.data(), sizeof(std::complex<double>) * vcc.size(), hipMemcpyHostToDevice));
    hipErrcheck(hipMemcpy(_eigenvalue, eigenvalue.data(), sizeof(double) * eigenvalue.size(), hipMemcpyHostToDevice));
}

template <>
void dngvx_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* d,
                                                                        const int nbase,
                                                                        const int ldh,
                                                                        std::complex<float>* hcc,
                                                                        std::complex<float>* scc,
                                                                        const int m,
                                                                        float* eigenvalue,
                                                                        std::complex<float>* vcc)
{
}

template <>
void dngvx_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* d,
                                                                         const int nbase,
                                                                         const int ldh,
                                                                         std::complex<double>* hcc,
                                                                         std::complex<double>* scc,
                                                                         const int m,
                                                                         double* eigenvalue,
                                                                         std::complex<double>* vcc)
{
}

#ifdef __LCAO
template <>
void dngvx_op<double, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* d,
                                                           const int nbase,
                                                           const int ldh,
                                                           double* hcc,
                                                           double* scc,
                                                           const int m,
                                                           double* eigenvalue,
                                                           double* vcc)
{
}
#endif // __LCAO

} // namespace hsolver