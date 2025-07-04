#include "source_estate/kernels/elecstate_op.h"

namespace elecstate{

template <typename FPTYPE>
struct elecstate_pw_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
                    const int& spin,
                    const int& nrxx,
                    const FPTYPE& w1,
                    FPTYPE** rho,
                    const std::complex<FPTYPE>* wfcr)
    {
      // for (int ir = 0; ir < nrxx; ir++)
      // {
      //   rho[spin][ir] += weight * norm(wfcr[ir]);
      // }
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ir = 0; ir < nrxx; ir++)
      {
        rho[spin][ir] += w1 * norm(wfcr[ir]);
      }
    }

    void operator()(const base_device::DEVICE_CPU* ctx,
                    const bool& DOMAG,
                    const bool& DOMAG_Z,
                    const int& nrxx,
                    const FPTYPE& w1,
                    FPTYPE** rho,
                    const std::complex<FPTYPE>* wfcr,
                    const std::complex<FPTYPE>* wfcr_another_spin)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ir = 0; ir < nrxx; ir++) {
          rho[0][ir] += w1 * (norm(wfcr[ir]) + norm(wfcr_another_spin[ir]));
      }
      // In this case, calculate the three components of the magnetization
      if (DOMAG)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int ir = 0; ir < nrxx; ir++) {
              rho[1][ir] += w1 * 2.0
                                          * (wfcr[ir].real() * wfcr_another_spin[ir].real()
                                             + wfcr[ir].imag() * wfcr_another_spin[ir].imag());
              rho[2][ir] += w1 * 2.0
                                          * (wfcr[ir].real() * wfcr_another_spin[ir].imag()
                                             - wfcr_another_spin[ir].real() * wfcr[ir].imag());
              rho[3][ir] += w1 * (norm(wfcr[ir]) - norm(wfcr_another_spin[ir]));
          }
      }
      else if (DOMAG_Z)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int ir = 0; ir < nrxx; ir++)
          {
              rho[1][ir] = 0;
              rho[2][ir] = 0;
              rho[3][ir] += w1 * (norm(wfcr[ir]) - norm(wfcr_another_spin[ir]));
          }
      }
      else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096/sizeof(FPTYPE))
#endif
          for (int is = 1; is < 4; is++)
          {
              for (int ir = 0; ir < nrxx; ir++)
                  rho[is][ir] = 0;
          }
      }
    }
};

template struct elecstate_pw_op<float, base_device::DEVICE_CPU>;
template struct elecstate_pw_op<double, base_device::DEVICE_CPU>;
}  // namespace elecstate