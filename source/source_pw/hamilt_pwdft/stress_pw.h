#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_source_pw_HAMILT_PWDFT_STRESS_PW_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_source_pw_HAMILT_PWDFT_STRESS_PW_H

#include "source_estate/elecstate.h"
#include "source_pw/hamilt_pwdft/VL_in_pw.h"
#include "stress_func.h"

template <typename FPTYPE, typename Device = base_device::DEVICE_CPU>
class Stress_PW : public Stress_Func<FPTYPE, Device>
{
  public:
    Stress_PW(const elecstate::ElecState* pelec_in) : pelec(pelec_in){};

    // calculate the stress in PW basis
    void cal_stress(ModuleBase::matrix& smearing_sigmatot,
                    UnitCell& ucell,
                    const pseudopot_cell_vl& locpp,
                    const pseudopot_cell_vnl& nlpp,
                    ModulePW::PW_Basis* rho_basis,
                    ModuleSymmetry::Symmetry* p_symm,
                    Structure_Factor* p_sf,
                    K_Vectors* p_kv,
                    ModulePW::PW_Basis_K* wfc_basis,
                    const psi::Psi<complex<FPTYPE>, Device>* d_psi_in = nullptr);

  protected:
    // call the vdw stress
    void stress_vdw(ModuleBase::matrix& smearing_sigma,
                    UnitCell& ucell); // force and stress calculated in vdw together.

    // the stress from the non-local pseudopotentials in uspp
    // which is due to the dependence of the Q function on the atomic position
    void stress_us(ModuleBase::matrix& sigma,
                   ModulePW::PW_Basis* rho_basis,
                   const pseudopot_cell_vnl& nlpp,
                   const UnitCell& ucell); // nonlocal part of uspp in PW basis

    // exx stress due to the scaling of the lattice vectors
    // see 10.1103/PhysRevB.73.125120 for details
    void stress_exx(ModuleBase::matrix& sigma,
                    const ModuleBase::matrix& wg,
                    ModulePW::PW_Basis* rho_basis,
                    ModulePW::PW_Basis_K* wfc_basis,
                    const K_Vectors* p_kv,
                    const psi::Psi<complex<FPTYPE>, Device>* d_psi_in,
                    const UnitCell& ucell); // exx stress in PW basis

    const elecstate::ElecState* pelec = nullptr;
};
#endif
