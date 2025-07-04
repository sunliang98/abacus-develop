#ifndef ELECSTATE_H
#define ELECSTATE_H

#include "fp_energy.h"
#include "source_cell/klist.h"
#include "source_estate/module_charge/charge.h"
#include "module_parameter/parameter.h"
#include "source_psi/psi.h"
#include "module_pot/potential_new.h"

namespace elecstate
{

class ElecState
{
  public:
    ElecState()
    {
    }
    ElecState(Charge* charge_in, ModulePW::PW_Basis* rhopw_in, ModulePW::PW_Basis_Big* bigpw_in)
    {
        this->charge = charge_in;
        this->charge->set_rhopw(rhopw_in);
        this->bigpw = bigpw_in;
        this->eferm.two_efermi = PARAM.globalv.two_fermi;
    }
    virtual ~ElecState()
    {
        if (this->pot != nullptr)
        {
            delete this->pot;
            this->pot = nullptr;
        }
    }
    void init_ks(Charge* chg_in, // pointer for class Charge
                 const K_Vectors* klist_in,
                 int nk_in, // number of k points
                 ModulePW::PW_Basis* rhopw_in,
                 const ModulePW::PW_Basis_Big* bigpw_in);

    // return current electronic density rho, as a input for constructing Hamiltonian
    virtual const double* getRho(int spin) const;

    // calculate electronic charge density on grid points or density matrix in real space
    // the consequence charge density rho saved into rho_out, preparing for charge mixing.
    virtual void psiToRho(const psi::Psi<std::complex<double>>& psi)
    {
        return;
    }
    virtual void psiToRho(const psi::Psi<double>& psi)
    {
        return;
    }
    // virtual void updateRhoK(const psi::Psi<std::complex<double>> &psi) = 0;
    // virtual void updateRhoK(const psi::Psi<double> &psi)=0
    virtual void cal_tau(const psi::Psi<std::complex<double>>& psi)
    {
        return;
    }
    virtual void cal_tau(const psi::Psi<double>& psi)
    {
        return;
    }
    virtual void cal_tau(const psi::Psi<std::complex<float>>& psi)
    {
        return;
    }

    // update charge density for next scf step
    // in this function, 1. input rho for construct Hamilt and 2. calculated rho from Psi will mix to 3. new charge
    // density rho among these rho,
    // 1. input rho would be store to file for restart
    // 2. calculated rho should be near with input rho when convergence has achieved
    // 3. new rho should be input rho for next scf step.
    virtual void getNewRho()
    {
        return;
    }



    // use occupied weights from INPUT and skip calculate_weights
    // mohan updated on 2024-06-08
    

    // if nupdown is not 0(TWO_EFERMI case),
    // nelec_spin will be fixed and weights will be constrained
    void init_nelec_spin();
    // used to record number of electrons per spin index
    // for NSPIN=2, it will record number of spin up and number of spin down
    // for NSPIN=4, it will record total number, magnetization for x, y, z direction
    std::vector<double> nelec_spin;

    virtual void print_psi(const psi::Psi<double>& psi_in, const int istep = -1)
    {
        return;
    }
    virtual void print_psi(const psi::Psi<std::complex<double>>& psi_in, const int istep = -1)
    {
        return;
    }

    /**
     * @brief Init rho_core, init rho, renormalize rho, init pot
     *
     * @param istep i-th step
     * @param ucell unit cell
     * @param strucfac structure factor
     * @param symm symmetry
     * @param wfcpw PW basis for wave function if needed
     */
    void init_scf(const int istep,
                  const UnitCell& ucell,
                  const Parallel_Grid& pgrid,
                  const ModuleBase::ComplexMatrix& strucfac,
                  const bool* numeric,
                  ModuleSymmetry::Symmetry& symm,
                  const void* wfcpw = nullptr);
    std::string classname = "elecstate";

    int iter = 0;                                  ///< scf iteration
    double omega = 0.0;                            ///< volume
    Potential* pot = nullptr;                      ///< pointer to potential
    Charge* charge = nullptr;                      ///< pointer to charge density
    const K_Vectors* klist = nullptr;              ///< pointer to k points lists
    const ModulePW::PW_Basis_Big* bigpw = nullptr; ///< bigpw will be removed later

  public: // something aboud energies. See elecstate_energy.cpp
    void cal_bandgap();
    void cal_bandgap_updw();

    double cal_delta_eband(const UnitCell& ucell) const;
    double cal_delta_escf() const;

    ModuleBase::matrix vnew;
    bool vnew_exist = false;
    void cal_converged();
    void cal_energies(const int type);
    void set_exx(const double& Eexx);
    void set_exx(const std::complex<double>& Eexx);

    double get_hartree_energy();
    double get_etot_efield();
    double get_etot_gatefield();

    double get_solvent_model_Ael();
    double get_solvent_model_Acav();

    virtual double get_spin_constrain_energy()
    {
        return 0.0;
    }

    double get_dftu_energy();
    double get_local_pp_energy();

    fenergy f_en; ///< energies contribute to the total free energy
    efermi eferm; ///< fermi energies

    // below defines the bandgap:

    double bandgap = 0.0;    ///< bandgap = E_{lumo} - E_{homo}
    double bandgap_up = 0.0; ///< spin up bandgap
    double bandgap_dw = 0.0; ///< spin down bandgap

    ModuleBase::matrix ekb; ///< band energy at each k point, each band.
    ModuleBase::matrix wg;  ///< occupation weight for each k-point and band

  public:

    bool skip_weights = false;
};

} // namespace elecstate
#endif
