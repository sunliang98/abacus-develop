#include "esolver_ks_pw.h"

#include "module_base/global_variable.h"
#include "module_hamilt_pw/hamilt_pwdft/elecond.h"
#include "module_io/input_conv.h"
#include "module_io/nscf_band.h"
#include "module_io/output_log.h"
#include "module_io/write_dos_pw.h"
#include "module_io/write_istate_info.h"
#include "module_io/write_wfc_pw.h"

#include <iostream>

//--------------temporary----------------------------
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_general/module_ewald/H_Ewald_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/print_info.h"
//-----force-------------------
#include "module_hamilt_pw/hamilt_pwdft/forces.h"
//-----stress------------------
#include "module_hamilt_pw/hamilt_pwdft/stress_pw.h"
//---------------------------------------------------
#include "module_base/memory.h"
#include "module_base/module_device/device.h"
#include "module_elecstate/elecstate_pw.h"
#include "module_hamilt_general/module_vdw/vdw.h"
#include "module_hamilt_pw/hamilt_pwdft/hamilt_pw.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_hsolver/hsolver_pw.h"
#include "module_hsolver/kernels/dngvd_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_io/berryphase.h"
#include "module_io/cube_io.h"
#include "module_io/get_pchg_pw.h"
#include "module_io/numerical_basis.h"
#include "module_io/numerical_descriptor.h"
#include "module_io/to_wannier90_pw.h"
#include "module_io/winput.h"
#include "module_io/write_elecstat_pot.h"
#include "module_io/write_wfc_r.h"
#include "module_parameter/parameter.h"
#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif
#include "module_base/formatter.h"
#ifdef __MLKEDF
#include "module_hamilt_pw/hamilt_ofdft/ml_data.h"
#endif

#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::ESolver_KS_PW()
{
    this->classname = "ESolver_KS_PW";
    this->basisname = "PW";
    this->device = base_device::get_device_type<Device>(this->ctx);
#if ((defined __CUDA) || (defined __ROCM))
    if (this->device == base_device::GpuDevice)
    {
        hsolver::createGpuBlasHandle();
        hsolver::createGpuSolverHandle();
        container::kernels::createGpuBlasHandle();
        container::kernels::createGpuSolverHandle();
    }
#endif
}

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::~ESolver_KS_PW()
{

    // delete Hamilt
    this->deallocate_hamilt();

    if (this->pelec != nullptr)
    {
        delete reinterpret_cast<elecstate::ElecStatePW<T, Device>*>(this->pelec);
        this->pelec = nullptr;
    }

    if (this->device == base_device::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        hsolver::destoryBLAShandle();
        hsolver::destroyGpuSolverHandle();
        container::kernels::destroyGpuBlasHandle();
        container::kernels::destroyGpuSolverHandle();
#endif
        delete reinterpret_cast<psi::Psi<T, Device>*>(this->kspw_psi);
    }
    
    if (PARAM.inp.precision == "single")
    {
        delete reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->__kspw_psi);
    }

    delete this->psi;
    delete this->p_wf_init;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_all_runners(const Input_para& inp, UnitCell& ucell)
{
    // 1) call before_all_runners() of ESolver_KS
    ESolver_KS<T, Device>::before_all_runners(inp, ucell);

    // 3) initialize ElecState,
    if (this->pelec == nullptr)
    {
        this->pelec = new elecstate::ElecStatePW<T, Device>(this->pw_wfc,
                                                            &(this->chr),
                                                            &(this->kv),
                                                            &ucell,
                                                            &GlobalC::ppcell,
                                                            this->pw_rhod,
                                                            this->pw_rho,
                                                            this->pw_big);
    }

    //! 4) inititlize the charge density.
    this->pelec->charge->allocate(PARAM.inp.nspin);

    //! 5) set the cell volume variable in pelec
    this->pelec->omega = ucell.omega;

    //! 6) initialize the potential.
    if (this->pelec->pot == nullptr)
    {
        this->pelec->pot = new elecstate::Potential(this->pw_rhod,
                                                    this->pw_rho,
                                                    &ucell,
                                                    &GlobalC::ppcell.vloc,
                                                    &(this->sf),
                                                    &(this->pelec->f_en.etxc),
                                                    &(this->pelec->f_en.vtxc));
    }

    //! 7) prepare some parameters for electronic wave functions initilization
    this->p_wf_init = new psi::WFInit<T, Device>(PARAM.inp.init_wfc,
                                                 PARAM.inp.ks_solver,
                                                 PARAM.inp.basis_type,
                                                 PARAM.inp.psi_initializer,
                                                 &this->wf,
                                                 this->pw_wfc);
    this->p_wf_init->prepare_init(&(this->sf),
                                  &ucell,
                                  1,
#ifdef __MPI
                                  &GlobalC::Pkpoints,
                                  GlobalV::MY_RANK,
#endif
                                  &GlobalC::ppcell);

    //! 8) setup global classes
    this->Init_GlobalC(inp, ucell, GlobalC::ppcell);

    //! 9) setup occupations
    if (PARAM.inp.ocp)
    {
        this->pelec->fixed_weights(PARAM.inp.ocp_kb, PARAM.inp.nbands, PARAM.inp.nelec);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_scf(const int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "before_scf");

    if (GlobalC::ucell.cell_parameter_updated)
    {
        this->init_after_vc(PARAM.inp, GlobalC::ucell);
    }
    if (GlobalC::ucell.ionic_position_updated)
    {
        this->CE.update_all_dis(GlobalC::ucell);
        this->CE.extrapolate_charge(
#ifdef __MPI
            &(GlobalC::Pgrid),
#endif
            GlobalC::ucell,
            this->pelec->charge,
            &this->sf,
            GlobalV::ofs_running,
            GlobalV::ofs_warning);
    }

    // init Hamilt, this should be allocated before each scf loop
    // Operators in HamiltPW should be reallocated once cell changed
    // delete Hamilt if not first scf
    this->deallocate_hamilt();

    // allocate HamiltPW
    this->allocate_hamilt();

    //----------------------------------------------------------
    // about vdw, jiyy add vdwd3 and linpz add vdwd2
    //----------------------------------------------------------
    auto vdw_solver = vdw::make_vdw(GlobalC::ucell, PARAM.inp);
    if (vdw_solver != nullptr)
    {
        this->pelec->f_en.evdw = vdw_solver->get_energy();
    }

    // calculate ewald energy
    if (!PARAM.inp.test_skip_ewald)
    {
        this->pelec->f_en.ewald_energy = H_Ewald_pw::compute_ewald(GlobalC::ucell, this->pw_rhod, this->sf.strucFac);
    }

    //! cal_ux should be called before init_scf because
    //! the direction of ux is used in noncoline_rho
    if (PARAM.inp.nspin == 4)
    {
        GlobalC::ucell.cal_ux();
    }

    //! calculate the total local pseudopotential in real space
    this->pelec->init_scf(istep, this->sf.strucFac, GlobalC::ucell.symm, (void*)this->pw_wfc);

    //! output the initial charge density
    if (PARAM.inp.out_chg[0] == 2)
    {
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            std::stringstream ss;
            ss << PARAM.globalv.global_out_dir << "SPIN" << is + 1 << "_CHG_INI.cube";
            ModuleIO::write_cube(
#ifdef __MPI
                this->pw_big->bz,
                this->pw_big->nbz,
                this->pw_rhod->nplane,
                this->pw_rhod->startz_current,
#endif
                this->pelec->charge->rho[is],
                is,
                PARAM.inp.nspin,
                istep,
                ss.str(),
                this->pw_rhod->nx,
                this->pw_rhod->ny,
                this->pw_rhod->nz,
                this->pelec->eferm.ef,
                &(GlobalC::ucell));
        }
    }

    //! output total local potential of the initial charge density
    if (PARAM.inp.out_pot == 3)
    {
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            std::stringstream ss;
            ss << PARAM.globalv.global_out_dir << "SPIN" << is + 1 << "_POT_INI.cube";
            ModuleIO::write_cube(
#ifdef __MPI
                this->pw_big->bz,
                this->pw_big->nbz,
                this->pw_rhod->nplane,
                this->pw_rhod->startz_current,
#endif
                this->pelec->pot->get_effective_v(is),
                is,
                PARAM.inp.nspin,
                istep,
                ss.str(),
                this->pw_rhod->nx,
                this->pw_rhod->ny,
                this->pw_rhod->nz,
                0.0, // efermi
                &(GlobalC::ucell),
                11, // precsion
                0); // out_fermi
        }
    }

    //! Symmetry_rho should behind init_scf, because charge should be
    //! initialized first. liuyu comment: Symmetry_rho should be located between
    //! init_rho and v_of_rho?
    Symmetry_rho srho;
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        srho.begin(is, *(this->pelec->charge), this->pw_rhod, GlobalC::ucell.symm);
    }

    // liuyu move here 2023-10-09
    // D in uspp need vloc, thus behind init_scf()
    // calculate the effective coefficient matrix for non-local pseudopotential
    // projectors
    ModuleBase::matrix veff = this->pelec->pot->get_effective_v();

    GlobalC::ppcell.cal_effective_D(veff, this->pw_rhod, GlobalC::ucell);

    // after init_rho (in pelec->init_scf), we have rho now.
    // before hamilt2density, we update Hk and initialize psi

    // before_scf function will be called everytime before scf. However, once
    // atomic coordinates changed, structure factor will change, therefore all
    // atomwise properties will change. So we need to reinitialize psi every
    // time before scf. But for random wavefunction, we dont, because random
    // wavefunction is not related to atomic coordinates. What the old strategy
    // does is only to initialize for once...
    if (((PARAM.inp.init_wfc == "random") && (istep == 0)) || (PARAM.inp.init_wfc != "random"))
    {
        this->p_wf_init->initialize_psi(this->psi, this->kspw_psi, this->p_hamilt, GlobalV::ofs_running);
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_init(const int istep, const int iter)
{
    if (iter == 1)
    {
        this->p_chgmix->init_mixing();
        this->p_chgmix->mixing_restart_step = PARAM.inp.scf_nmax + 1;
    }
    // for mixing restart
    if (iter == this->p_chgmix->mixing_restart_step && PARAM.inp.mixing_restart > 0.0)
    {
        this->p_chgmix->init_mixing();
    }
    // mohan move harris functional to here, 2012-06-05
    // use 'rho(in)' and 'v_h and v_xc'(in)
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband();

    //(2) save change density as previous charge,
    // prepared fox mixing.
    if (GlobalV::MY_STOGROUP == 0)
    {
        this->pelec->charge->save_rho_before_sum_band();
    }
}

// Temporary, it should be replaced by hsolver later.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2density(const int istep, const int iter, const double ethr)
{
    ModuleBase::timer::tick("ESolver_KS_PW", "hamilt2density");

    {
        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        // choose if psi should be diag in subspace
        // be careful that istep start from 0 and iter start from 1
        // if (iter == 1)
        hsolver::DiagoIterAssist<T, Device>::need_subspace = ((istep == 0 || istep == 1) && iter == 1) ? false : true;
        
        hsolver::DiagoIterAssist<T, Device>::SCF_ITER = iter;
        hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR = ethr;
        hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX = PARAM.inp.pw_diag_nmax;
        
        hsolver::HSolverPW<T, Device> hsolver_pw_obj(this->pw_wfc, 
                                                     &this->wf, 
                                                     
                                                     PARAM.inp.calculation,
                                                     PARAM.inp.basis_type,
                                                     PARAM.inp.ks_solver,
                                                     PARAM.inp.use_paw,
                                                     PARAM.globalv.use_uspp,
                                                     PARAM.inp.nspin,
                                                     
                                                     hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,

                                                     hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                     this->init_psi);
        
        hsolver_pw_obj.solve(this->p_hamilt,
                           this->kspw_psi[0],
                           this->pelec,
                           this->pelec->ekb.c,
                           GlobalV::RANK_IN_POOL,
                           GlobalV::NPROC_IN_POOL,
                           false);
      
        this->init_psi = true;

        if (PARAM.inp.out_bandgap)
        {
            if (!PARAM.globalv.two_fermi)
            {
                this->pelec->cal_bandgap();
            }
            else
            {
                this->pelec->cal_bandgap_updw();
            }
        }
    }

    // calculate the delta_harris energy
    // according to new charge density.
    // mohan add 2009-01-23
    this->pelec->cal_energies(1);

    Symmetry_rho srho;
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        srho.begin(is, *(this->pelec->charge), this->pw_rhod, GlobalC::ucell.symm);
    }

    // compute magnetization, only for LSDA(spin==2)
    GlobalC::ucell.magnet.compute_magnetization(this->pelec->charge->nrxx,
                                                this->pelec->charge->nxyz,
                                                this->pelec->charge->rho,
                                                this->pelec->nelec_spin.data());

    // deband is calculated from "output" charge density calculated
    // in sum_band
    // need 'rho(out)' and 'vr (v_h(in) and v_xc(in))'
    this->pelec->f_en.deband = this->pelec->cal_delta_eband();

    ModuleBase::timer::tick("ESolver_KS_PW", "hamilt2density");
}

// Temporary, it should be rewritten with Hamilt class.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::update_pot(const int istep, const int iter)
{
    if (!this->conv_esolver)
    {
        if (PARAM.inp.nspin == 4)
        {
            GlobalC::ucell.cal_ux();
        }
        this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);
        this->pelec->f_en.descf = this->pelec->cal_delta_escf();
    }
    else
    {
        this->pelec->cal_converged();
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_finish(int& iter)
{
    // call iter_finish() of ESolver_KS
    ESolver_KS<T, Device>::iter_finish(iter);

    // liuyu 2023-10-24
    // D in uspp need vloc, thus needs update when veff updated
    // calculate the effective coefficient matrix for non-local pseudopotential
    // projectors
    if (PARAM.globalv.use_uspp)
    {
        ModuleBase::matrix veff = this->pelec->pot->get_effective_v();
        GlobalC::ppcell.cal_effective_D(veff, this->pw_rhod, GlobalC::ucell);
    }

    if (this->out_freq_elec && iter % this->out_freq_elec == 0)
    {
        if (PARAM.inp.out_chg[0] > 0)
        {
            for (int is = 0; is < PARAM.inp.nspin; is++)
            {
                double* data = nullptr;
                if (PARAM.inp.dm_to_rho)
                {
                    data = this->pelec->charge->rho[is];
                }
                else
                {
                    data = this->pelec->charge->rho_save[is];
                }
                std::string fn = PARAM.globalv.global_out_dir + "/tmp_SPIN" + std::to_string(is + 1) + "_CHG.cube";
                ModuleIO::write_cube(
#ifdef __MPI
                    this->pw_big->bz,
                    this->pw_big->nbz,
                    this->pw_rhod->nplane,
                    this->pw_rhod->startz_current,
#endif
                    data,
                    is,
                    PARAM.inp.nspin,
                    0,
                    fn,
                    this->pw_rhod->nx,
                    this->pw_rhod->ny,
                    this->pw_rhod->nz,
                    this->pelec->eferm.get_efval(is),
                    &(GlobalC::ucell),
                    3,
                    1);
                if (XC_Functional::get_func_type() == 3 || XC_Functional::get_func_type() == 5)
                {
                    fn = PARAM.globalv.global_out_dir + "/tmp_SPIN" + std::to_string(is + 1) + "_TAU.cube";
                    ModuleIO::write_cube(
#ifdef __MPI
                        this->pw_big->bz,
                        this->pw_big->nbz,
                        this->pw_rhod->nplane,
                        this->pw_rhod->startz_current,
#endif
                        this->pelec->charge->kin_r_save[is],
                        is,
                        PARAM.inp.nspin,
                        0,
                        fn,
                        this->pw_rhod->nx,
                        this->pw_rhod->ny,
                        this->pw_rhod->nz,
                        this->pelec->eferm.get_efval(is),
                        &(GlobalC::ucell));
                }
            }
        }
        // output wavefunctions
        if (this->wf.out_wfc_pw == 1 || this->wf.out_wfc_pw == 2)
        {
            std::stringstream ssw;
            ssw << PARAM.globalv.global_out_dir << "WAVEFUNC";
            // mohan update 2011-02-21
            // qianrui update 2020-10-17
            ModuleIO::write_wfc_pw(ssw.str(), this->psi[0], this->kv, this->pw_wfc);
            // ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running,"write wave
            // functions into file WAVEFUNC.dat");
        }
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_scf(const int istep)
{
    // 1) calculate the kinetic energy density tau, sunliang 2024-09-18
    if (PARAM.inp.out_elf[0] > 0)
    {
        this->pelec->cal_tau(*(this->psi));
    }

    // 2) call after_scf() of ESolver_KS
    ESolver_KS<T, Device>::after_scf(istep);

    // 3) output wavefunctions
    if (this->wf.out_wfc_pw == 1 || this->wf.out_wfc_pw == 2)
    {
        std::stringstream ssw;
        ssw << PARAM.globalv.global_out_dir << "WAVEFUNC";
        ModuleIO::write_wfc_pw(ssw.str(), this->psi[0], this->kv, this->pw_wfc);
    }

    if (this->device == base_device::GpuDevice)
    {
        castmem_2d_d2h_op()(this->psi[0].get_device(),
                            this->kspw_psi[0].get_device(),
                            this->psi[0].get_pointer() - this->psi[0].get_psi_bias(),
                            this->kspw_psi[0].get_pointer() - this->kspw_psi[0].get_psi_bias(),
                            this->psi[0].size());
    }

    // Calculate band-decomposed (partial) charge density
    const std::vector<int> bands_to_print = PARAM.inp.bands_to_print;
    if (bands_to_print.size() > 0)
    {
        ModuleIO::get_pchg_pw(bands_to_print,
                              this->kspw_psi->get_nbands(),
                              PARAM.inp.nspin,
                              this->pw_rhod->nx,
                              this->pw_rhod->ny,
                              this->pw_rhod->nz,
                              this->pw_rhod->nxyz,
                              this->kv.get_nks(),
                              this->kv.isk,
                              this->kv.wk,
                              this->pw_big->bz,
                              this->pw_big->nbz,
                              this->pelec->charge->ngmc,
                              &GlobalC::ucell,
                              this->psi,
                              this->pw_rhod,
                              this->pw_wfc,
                              this->ctx,
                              GlobalC::Pgrid,
                              PARAM.globalv.global_out_dir,
                              PARAM.inp.if_separate_k);
    }
}

template <typename T, typename Device>
double ESolver_KS_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_force(ModuleBase::matrix& force)
{
    Forces<double, Device> ff(GlobalC::ucell.nat);
    if (this->__kspw_psi != nullptr && PARAM.inp.precision == "single")
    {
        delete reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->__kspw_psi);
    }

    // Refresh __kspw_psi
    this->__kspw_psi = PARAM.inp.precision == "single"
                           ? new psi::Psi<std::complex<double>, Device>(this->kspw_psi[0])
                           : reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->kspw_psi);

    // Calculate forces
    ff.cal_force(force,
                 *this->pelec,
                 this->pw_rhod,
                 &GlobalC::ucell.symm,
                 &this->sf,
                 &this->kv,
                 this->pw_wfc,
                 this->__kspw_psi);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_stress(ModuleBase::matrix& stress)
{
    Stress_PW<double, Device> ss(this->pelec);
    if (this->__kspw_psi != nullptr && PARAM.inp.precision == "single")
    {
        delete reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->__kspw_psi);
    }

    // Refresh __kspw_psi
    this->__kspw_psi = PARAM.inp.precision == "single"
                           ? new psi::Psi<std::complex<double>, Device>(this->kspw_psi[0])
                           : reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->kspw_psi);
    ss.cal_stress(stress,
                  GlobalC::ucell,
                  &GlobalC::ppcell,
                  this->pw_rhod,
                  &GlobalC::ucell.symm,
                  &this->sf,
                  &this->kv,
                  this->pw_wfc,
                  this->psi,
                  this->__kspw_psi);

    // external stress
    double unit_transform = 0.0;
    unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    double external_stress[3] = {PARAM.inp.press1, PARAM.inp.press2, PARAM.inp.press3};
    for (int i = 0; i < 3; i++)
    {
        stress(i, i) -= external_stress[i] / unit_transform;
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_all_runners()
{
    GlobalV::ofs_running << "\n\n --------------------------------------------" << std::endl;
    GlobalV::ofs_running << std::setprecision(16);
    GlobalV::ofs_running << " !FINAL_ETOT_IS " << this->pelec->f_en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
    GlobalV::ofs_running << " --------------------------------------------\n\n" << std::endl;

    if (PARAM.inp.out_dos != 0 || PARAM.inp.out_band[0] != 0)
    {
        GlobalV::ofs_running << "\n\n\n\n";
        GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                                ">>>>>>>>>>>>>>>>>>>>>>>>>"
                             << std::endl;
        GlobalV::ofs_running << " |                                            "
                                "                        |"
                             << std::endl;
        GlobalV::ofs_running << " | Post-processing of data:                   "
                                "                        |"
                             << std::endl;
        GlobalV::ofs_running << " | DOS (density of states) and bands will be "
                                "output here.             |"
                             << std::endl;
        GlobalV::ofs_running << " | If atomic orbitals are used, Mulliken "
                                "charge analysis can be done. |"
                             << std::endl;
        GlobalV::ofs_running << " | Also the .bxsf file containing fermi "
                                "surface information can be    |"
                             << std::endl;
        GlobalV::ofs_running << " | done here.                                 "
                                "                        |"
                             << std::endl;
        GlobalV::ofs_running << " |                                            "
                                "                        |"
                             << std::endl;
        GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                                "<<<<<<<<<<<<<<<<<<<<<<<<<"
                             << std::endl;
        GlobalV::ofs_running << "\n\n\n\n";
    }

    int nspin0 = 1;
    if (PARAM.inp.nspin == 2)
    {
        nspin0 = 2;
    }
    //! print occupation in istate.info
    ModuleIO::write_istate_info(this->pelec->ekb, this->pelec->wg, this->kv, &(GlobalC::Pkpoints));

    //! compute density of states
    if (PARAM.inp.out_dos)
    {
        ModuleIO::write_dos_pw(this->pelec->ekb,
                               this->pelec->wg,
                               this->kv,
                               PARAM.inp.dos_edelta_ev,
                               PARAM.inp.dos_scale,
                               PARAM.inp.dos_sigma);

        if (nspin0 == 1)
        {
            GlobalV::ofs_running << " Fermi energy is " << this->pelec->eferm.ef << " Rydberg" << std::endl;
        }
        else if (nspin0 == 2)
        {
            GlobalV::ofs_running << " Fermi energy (spin = 1) is " << this->pelec->eferm.ef_up << " Rydberg"
                                 << std::endl;
            GlobalV::ofs_running << " Fermi energy (spin = 2) is " << this->pelec->eferm.ef_dw << " Rydberg"
                                 << std::endl;
        }
    }

    if (PARAM.inp.out_band[0]) // pengfei 2014-10-13
    {
        for (int is = 0; is < nspin0; is++)
        {
            std::stringstream ss2;
            ss2 << PARAM.globalv.global_out_dir << "BANDS_" << is + 1 << ".dat";
            GlobalV::ofs_running << "\n Output bands in file: " << ss2.str() << std::endl;
            ModuleIO::nscf_band(is,
                                ss2.str(),
                                PARAM.inp.nbands,
                                0.0,
                                PARAM.inp.out_band[1],
                                this->pelec->ekb,
                                this->kv,
                                &(GlobalC::Pkpoints));
        }
    }

    if (PARAM.inp.basis_type == "pw" && winput::out_spillage) // xiaohui add 2013-09-01
    {
        //  calculate spillage value.

        // ! Print out overlap before spillage optimization to generate atomic
        // orbitals
        if (winput::out_spillage <= 2)
        {
            for (int i = 0; i < PARAM.inp.bessel_nao_rcuts.size(); i++)
            {
                if (GlobalV::MY_RANK == 0)
                {
                    std::cout << "update value: bessel_nao_rcut <- " << std::fixed << PARAM.inp.bessel_nao_rcuts[i]
                              << " a.u." << std::endl;
                }
                Numerical_Basis numerical_basis;
                numerical_basis.output_overlap(this->psi[0], this->sf, this->kv, this->pw_wfc, GlobalC::ucell, i);
            }
            ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "BASIS OVERLAP (Q and S) GENERATION.");
        }
    }

    //! Print out wave functions in real space
    if (this->wf.out_wfc_r == 1) // Peize Lin add 2021.11.21
    {
        ModuleIO::write_psi_r_1(this->psi[0], this->pw_wfc, "wfc_realspace", true, this->kv);
    }

    //! Use Kubo-Greenwood method to compute conductivities
    if (PARAM.inp.cal_cond)
    {
        EleCond elec_cond(&GlobalC::ucell, &this->kv, this->pelec, this->pw_wfc, this->psi, &GlobalC::ppcell);
        elec_cond.KG(PARAM.inp.cond_smear,
                     PARAM.inp.cond_fwhm,
                     PARAM.inp.cond_wcut,
                     PARAM.inp.cond_dw,
                     PARAM.inp.cond_dt,
                     PARAM.inp.cond_nonlocal,
                     this->pelec->wg);
    }

#ifdef __MLKEDF
    // generate training data for ML-KEDF
    if(PARAM.inp.of_ml_gene_data == 1)
    {
        this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);

        ML_data ml_data;
        ml_data.set_para(this->pelec->charge->nrxx, PARAM.inp.nelec, PARAM.inp.of_tf_weight, PARAM.inp.of_vw_weight,
                            PARAM.inp.of_ml_chi_p, PARAM.inp.of_ml_chi_q, PARAM.inp.of_ml_chi_xi, PARAM.inp.of_ml_chi_pnl, PARAM.inp.of_ml_chi_qnl,
                            PARAM.inp.of_ml_nkernel, PARAM.inp.of_ml_kernel, PARAM.inp.of_ml_kernel_scaling, PARAM.inp.of_ml_yukawa_alpha, PARAM.inp.of_ml_kernel_file, this->pw_rho);
        ml_data.generateTrainData_KS(this->kspw_psi, this->pelec, this->pw_wfc, this->pw_rho, this->pelec->pot->get_effective_v(0));
    }
#endif
}

template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver
