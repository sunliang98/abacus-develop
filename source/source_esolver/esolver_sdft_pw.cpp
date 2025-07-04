#include "esolver_sdft_pw.h"

#include "source_base/global_variable.h"
#include "source_base/memory.h"
#include "source_estate/module_charge/symmetry_rho.h"
#include "source_pw/hamilt_stodft/sto_dos.h"
#include "source_pw/hamilt_stodft/sto_elecond.h"
#include "source_pw/hamilt_stodft/sto_forces.h"
#include "source_pw/hamilt_stodft/sto_stress_pw.h"
#include "source_hsolver/diago_iter_assist.h"
#include "module_parameter/parameter.h"

#include <algorithm>
#include <fstream>

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_SDFT_PW<T, Device>::ESolver_SDFT_PW()
    : stoche(PARAM.inp.nche_sto, PARAM.inp.method_sto, PARAM.inp.emax_sto, PARAM.inp.emin_sto)
{
    this->classname = "ESolver_SDFT_PW";
    this->basisname = "PW";
}

template <typename T, typename Device>
ESolver_SDFT_PW<T, Device>::~ESolver_SDFT_PW()
{
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    // 1) initialize parameters from int Input class
    this->nche_sto = inp.nche_sto;
    this->method_sto = inp.method_sto;

    // 2) run "before_all_runners" in ESolver_KS
    ESolver_KS_PW<T, Device>::before_all_runners(ucell, inp);

    // 3) initialize the stochastic wave functions
    this->stowf.init(&this->kv, this->pw_wfc->npwk_max);
    if (inp.nbands_sto != 0)
    {
        if (inp.initsto_ecut < inp.ecutwfc)
        {
            this->stowf.init_sto_orbitals(inp.seed_sto);
        }
        else
        {
            this->stowf.init_sto_orbitals_Ecut(inp.seed_sto, this->kv, *this->pw_wfc, inp.initsto_ecut);
        }
    }
    else
    {
        this->stowf.init_com_orbitals();
    }
    if (this->method_sto == 2)
    {
        this->stowf.allocate_chiallorder(this->nche_sto);
    }
    this->stowf.sync_chi0();

    // 4) allocate spaces for \sqrt(f(H))|chi> and |\tilde{chi}>
    size_t size = stowf.chi0->size();
    this->stowf.shchi
        = new psi::Psi<T, Device>(this->kv.get_nks(), 
                                  this->stowf.nchip_max, 
                                  this->pw_wfc->npwk_max, 
                                  this->kv.ngk,
                                  true);
    ModuleBase::Memory::record("SDFT::shchi", size * sizeof(T));

    if (PARAM.inp.nbands > 0)
    {
        this->stowf.chiortho
            = new psi::Psi<T, Device>(this->kv.get_nks(), 
                                      this->stowf.nchip_max, 
                                      this->pw_wfc->npwk_max, 
                                      this->kv.ngk, true);
        ModuleBase::Memory::record("SDFT::chiortho", size * sizeof(T));
    }

    return;
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_SDFT_PW", "before_scf");
    ModuleBase::timer::tick("ESolver_SDFT_PW", "before_scf");

    ESolver_KS_PW<T, Device>::before_scf(ucell, istep);
    delete reinterpret_cast<hamilt::HamiltPW<double>*>(this->p_hamilt);
    this->p_hamilt = new hamilt::HamiltSdftPW<T, Device>(this->pelec->pot,
                                                         this->pw_wfc,
                                                         &this->kv,
                                                         &this->ppcell,
                                                         &ucell, 
                                                         PARAM.globalv.npol,
                                                         &this->stoche.emin_sto,
                                                         &this->stoche.emax_sto);
    this->p_hamilt_sto = static_cast<hamilt::HamiltSdftPW<T, Device>*>(this->p_hamilt);

    if (istep > 0 && PARAM.inp.nbands_sto != 0 && PARAM.inp.initsto_freq > 0 && istep % PARAM.inp.initsto_freq == 0)
    {
        this->stowf.update_sto_orbitals(PARAM.inp.seed_sto);
    }

    ModuleBase::timer::tick("ESolver_SDFT_PW", "before_scf");
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    // call iter_finish() of ESolver_KS
    ESolver_KS<T, Device>::iter_finish(ucell, istep, iter, conv_esolver);
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::after_scf(UnitCell& ucell, const int istep, const bool conv_esolver)
{
    ModuleBase::TITLE("ESolver_SDFT_PW", "after_scf");
    ModuleBase::timer::tick("ESolver_SDFT_PW", "after_scf");

    // 1) call after_scf() of ESolver_KS_PW
    ESolver_KS_PW<T, Device>::after_scf(ucell, istep, conv_esolver);

    ModuleBase::timer::tick("ESolver_SDFT_PW", "after_scf");
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::hamilt2rho_single(UnitCell& ucell, int istep, int iter, double ethr)
{
    ModuleBase::TITLE("ESolver_SDFT_PW", "hamilt2rho");
    ModuleBase::timer::tick("ESolver_SDFT_PW", "hamilt2rho");

    // reset energy
    this->pelec->f_en.eband = 0.0;
    this->pelec->f_en.demet = 0.0;
    // choose if psi should be diag in subspace
    // be careful that istep start from 0 and iter start from 1
    if (istep == 0 && iter == 1 || PARAM.inp.calculation == "nscf")
    {
        hsolver::DiagoIterAssist<T, Device>::need_subspace = false;
    }
    else
    {
        hsolver::DiagoIterAssist<T, Device>::need_subspace = true;
    }

    bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;
    hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR = ethr;
    hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX = PARAM.inp.pw_diag_nmax;

    // hsolver only exists in this function
    hsolver::HSolverPW_SDFT<T, Device> hsolver_pw_sdft_obj(&this->kv,
                                                           this->pw_wfc,
                                                           this->stowf,
                                                           this->stoche,
                                                           this->p_hamilt_sto,
                                                           PARAM.inp.calculation,
                                                           PARAM.inp.basis_type,
                                                           PARAM.inp.ks_solver,
                                                           false,
                                                           PARAM.globalv.use_uspp,
                                                           PARAM.inp.nspin,
                                                           hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                           hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                           hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                           hsolver::DiagoIterAssist<T, Device>::need_subspace);

    hsolver_pw_sdft_obj.solve(ucell,
                              this->p_hamilt,
                              this->kspw_psi[0],
                              this->psi[0],
                              this->pelec,
                              this->pw_wfc,
                              this->stowf,
                              istep,
                              iter,
                              skip_charge);

    // set_diagethr need it
    this->esolver_KS_ne = hsolver_pw_sdft_obj.stoiter.KS_ne;

    if (PARAM.globalv.ks_run)
    {
        Symmetry_rho srho;
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            srho.begin(is, this->chr, this->pw_rho, ucell.symm);
        }
        this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);
    }
    else
    {
#ifdef __MPI
        if (ModuleSymmetry::Symmetry::symm_flag == 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif
    }
#ifdef __MPI
    MPI_Bcast(&(this->pelec->f_en.deband), 1, MPI_DOUBLE, 0, BP_WORLD);
#endif
    ModuleBase::timer::tick("ESolver_SDFT_PW", "hamilt2rho");
}

template <typename T, typename Device>
double ESolver_SDFT_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    Sto_Forces<double, Device> ff(ucell.nat);

    ff.cal_stoforce(force,
                    *this->pelec,
                    this->pw_rho,
                    &ucell.symm,
                    &this->sf,
                    &this->kv,
                    this->pw_wfc,
                    this->locpp,
                    this->ppcell,
                    ucell,
                    *this->kspw_psi,
                    this->stowf);
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    Sto_Stress_PW<double, Device> ss;
    ss.cal_stress(stress,
                  *this->pelec,
                  this->pw_rho,
                  &ucell.symm,
                  &this->sf,
                  &this->kv,
                  this->pw_wfc,
                  *this->kspw_psi,
                  this->stowf,
                  &this->chr,
                  &this->locpp,
                  &this->ppcell,
                  ucell);
}

template <typename T, typename Device>
void ESolver_SDFT_PW<T, Device>::after_all_runners(UnitCell& ucell)
{
    // 1) write down etot and eigenvalues (for MDFT) information
    ESolver_FP::after_all_runners(ucell);

    // 2) release memory
    if (this->method_sto == 2)
    {
        stowf.clean_chiallorder(); // release lots of memories
    }

    // 3) write down DOS
    if (PARAM.inp.out_dos)
    {
        if(!std::is_same<T, std::complex<double>>::value || !std::is_same<Device, base_device::DEVICE_CPU>::value)
        {
            ModuleBase::WARNING_QUIT("ESolver_SDFT_PW", "DOS does not support complex float or GPU yet.");
        }
        Sto_DOS<Real, Device> sto_dos(
            this->pw_wfc,
            &this->kv,
            this->pelec,
            reinterpret_cast<psi::Psi<std::complex<double>>*>(this->psi),
            reinterpret_cast<hamilt::Hamilt<std::complex<double>>*>(this->p_hamilt),
            this->stoche,
            reinterpret_cast<Stochastic_WF<std::complex<double>, base_device::DEVICE_CPU>*>(&stowf));
        sto_dos.decide_param(PARAM.inp.dos_nche,
                             PARAM.inp.emin_sto,
                             PARAM.inp.emax_sto,
                             PARAM.globalv.dos_setemin,
                             PARAM.globalv.dos_setemax,
                             PARAM.inp.dos_emin_ev,
                             PARAM.inp.dos_emax_ev,
                             PARAM.inp.dos_scale);
        sto_dos.caldos(PARAM.inp.dos_sigma, PARAM.inp.dos_edelta_ev, PARAM.inp.npart_sto);
    }

    // 4) sKG cost memory, and it should be placed at the end of the program
    if (PARAM.inp.cal_cond)
    {
        Sto_EleCond<Real, Device> sto_elecond(&ucell,
                                              &this->kv,
                                              this->pelec,
                                              this->pw_wfc,
                                              this->kspw_psi,
                                              &this->ppcell,
                                              this->p_hamilt,
                                              this->stoche,
                                              &stowf);
        sto_elecond.decide_nche(PARAM.inp.cond_dt, 1e-8, this->nche_sto, PARAM.inp.emin_sto, PARAM.inp.emax_sto);
        sto_elecond.sKG(PARAM.inp.cond_smear,
                        PARAM.inp.cond_fwhm,
                        PARAM.inp.cond_wcut,
                        PARAM.inp.cond_dw,
                        PARAM.inp.cond_dt,
                        PARAM.inp.cond_nonlocal,
                        PARAM.inp.npart_sto);
    }
}


// template class ESolver_SDFT_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_SDFT_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
// template class ESolver_SDFT_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_SDFT_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver
