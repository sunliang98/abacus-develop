#ifndef ELECSTATEPW_H
#define ELECSTATEPW_H

#include <source_base/macros.h>

#include "elecstate.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_estate/kernels/elecstate_op.h"
#include "source_pw/hamilt_pwdft/kernels/meta_op.h"
#include "source_base/kernels/math_kernel_op.h"

namespace elecstate
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class ElecStatePW : public ElecState
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    ElecStatePW(ModulePW::PW_Basis_K* wfc_basis_in,
                Charge* chg_in,
                K_Vectors* pkv_in,
                UnitCell* ucell_in,
                pseudopot_cell_vnl* ppcell_in,
                ModulePW::PW_Basis* rhodpw_in,
                ModulePW::PW_Basis* rhopw_in,
                ModulePW::PW_Basis_Big* bigpw_in);

    ~ElecStatePW();

    //! interface for HSolver to calculate rho from Psi
    virtual void psiToRho(const psi::Psi<T, Device>& psi);

    virtual void cal_tau(const psi::Psi<T, Device>& psi);

    //! calculate becsum for uspp
    void cal_becsum(const psi::Psi<T, Device>& psi);

    Real* becsum = nullptr;

    //! init rho_data and kin_r_data
    void init_rho_data();
    Real** rho = nullptr;   // [Device] [spin][nrxx] rho
    T** rhog = nullptr;     // [Device] [spin][nrxx] rhog
    Real** kin_r = nullptr; // [Device] [spin][nrxx] kin_r

    ModulePW::PW_Basis_K* basis = nullptr;

  protected:

    ModulePW::PW_Basis* rhopw_smooth = nullptr;

    UnitCell* ucell = nullptr;

    const pseudopot_cell_vnl* ppcell = nullptr;

    //! calculate electronic charge density on grid points or density matrix in real space
    //! the consequence charge density rho saved into rho_out, preparing for charge mixing.
    void updateRhoK(const psi::Psi<T, Device>& psi); // override;
    
    //! sum over all pools for rho and ebands
    void parallelK();
    
    //! calcualte rho for each k
    void rhoBandK(const psi::Psi<T, Device>& psi);

    //! add to the charge density in reciprocal space the part which is due to the US augmentation.
    void add_usrho(const psi::Psi<T, Device>& psi);

    //! Non-local pseudopotentials
    //! \sum_lm Q_lm(r) \sum_i <psi_i|beta_l><beta_m|psi_i> w_i
    void addusdens_g(const Real* becsum, T** rhog);

    Device * ctx = {};

    bool init_rho = false;

    mutable T* vkb = nullptr;

    Real* rho_data = nullptr;
    T* rhog_data = nullptr;
    Real* kin_r_data = nullptr;
    T* wfcr = nullptr; 
    T* wfcr_another_spin = nullptr;

  private:
    using meta_op = hamilt::meta_pw_op<Real, Device>;
    using elecstate_pw_op = elecstate::elecstate_pw_op<Real, Device>;

    using setmem_var_op = base_device::memory::set_memory_op<Real, Device>;
    using resmem_var_op = base_device::memory::resize_memory_op<Real, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<Real, Device>;
    using castmem_var_d2h_op = base_device::memory::cast_memory_op<double, Real, base_device::DEVICE_CPU, Device>;

    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;

    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
};

} // namespace elecstate

#endif
