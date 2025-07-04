#include "./elecstate_pw_sdft.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "module_parameter/parameter.h"
namespace elecstate
{

template <typename T, typename Device>
void ElecStatePW_SDFT<T, Device>::psiToRho(const psi::Psi<T, Device>& psi)
{
    ModuleBase::TITLE(this->classname, "psiToRho");
    ModuleBase::timer::tick(this->classname, "psiToRho");
    const int nspin = PARAM.inp.nspin;
    for (int is = 0; is < nspin; is++)
    {
        setmem_var_op()(this->rho[is], 0, this->charge->nrxx);
    }

    if (PARAM.globalv.ks_run)
    {
        for (int ik = 0; ik < psi.get_nk(); ++ik)
        {
            psi.fix_k(ik);
            this->updateRhoK(psi);
        }
        if (PARAM.inp.device == "gpu" || PARAM.inp.precision == "single") {
        for (int ii = 0; ii < nspin; ii++) {
            castmem_var_d2h_op()(this->charge->rho[ii], this->rho[ii], this->charge->nrxx);
        }
        }
        this->parallelK();
    }
    ModuleBase::timer::tick(this->classname, "psiToRho");
    return;
}

// template class ElecStatePW_SDFT<std::complex<float>, base_device::DEVICE_CPU>;
template class ElecStatePW_SDFT<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ElecStatePW_SDFT<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace elecstate