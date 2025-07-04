#include "esolver_ks_lcaopw.h"

#include "source_pw/module_pwdft/elecond.h"
#include "source_io/input_conv.h"
#include "source_io/output_log.h"

#include <iostream>

//--------------temporary----------------------------
#include "source_estate/module_charge/symmetry_rho.h"
#include "source_estate/occupy.h"
#include "source_hamilt/module_ewald/H_Ewald_pw.h"
#include "source_pw/module_pwdft/global.h"
#include "source_io/print_info.h"
//-----force-------------------
#include "source_pw/module_pwdft/forces.h"
//-----stress------------------
#include "source_pw/module_pwdft/stress_pw.h"
//---------------------------------------------------
#include "source_base/memory.h"
#include "source_estate/elecstate_pw.h"
#include "source_pw/module_pwdft/hamilt_lcaopw.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_hsolver/hsolver_lcaopw.h"
#include "source_hsolver/kernels/dngvd_op.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_io/berryphase.h"
#include "source_io/numerical_basis.h"
#include "source_io/numerical_descriptor.h"
#include "source_io/to_wannier90_pw.h"
#include "source_io/winput.h"
#include "source_io/write_elecstat_pot.h"
#include "module_parameter/parameter.h"

#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>
#include <sys/time.h>
#ifdef __LCAO
#include "source_io/write_vxc_lip.hpp"
#endif

namespace ModuleESolver
{

    template <typename T>
    ESolver_KS_LIP<T>::ESolver_KS_LIP()
    {
        this->classname = "ESolver_KS_LIP";
        this->basisname = "LIP";
    }
    template <typename T>
    ESolver_KS_LIP<T>::~ESolver_KS_LIP()
    {
        delete this->psi_local;
        // delete Hamilt
        this->deallocate_hamilt();
    }

    template <typename T>
    void ESolver_KS_LIP<T>::allocate_hamilt(const UnitCell& ucell)
    {
        this->p_hamilt = new hamilt::HamiltLIP<T>(this->pelec->pot, this->pw_wfc, &this->kv, &this->ppcell, &ucell
#ifdef __EXX
            , *this->exx_lip
#endif
        );
    }
    template <typename T>
    void ESolver_KS_LIP<T>::deallocate_hamilt()
    {
        if (this->p_hamilt != nullptr)
        {
            delete reinterpret_cast<hamilt::HamiltLIP<T>*>(this->p_hamilt);
            this->p_hamilt = nullptr;
        }
    }
    template <typename T>
    void ESolver_KS_LIP<T>::before_scf(UnitCell& ucell, const int istep)
    {
        ESolver_KS_PW<T>::before_scf(ucell, istep);
        this->p_psi_init->initialize_lcao_in_pw(this->psi_local, GlobalV::ofs_running);
    }

    template <typename T>
    void ESolver_KS_LIP<T>::before_all_runners(UnitCell& ucell, const Input_para& inp)
    {
        ESolver_KS_PW<T>::before_all_runners(ucell, inp);
        delete this->psi_local;
        this->psi_local = new psi::Psi<T>(this->psi->get_nk(),
                                          this->p_psi_init->psi_initer->nbands_start(),
                                          this->psi->get_nbasis(),
                                          this->kv.ngk,
                                          true);
#ifdef __EXX
        if (PARAM.inp.calculation == "scf" || PARAM.inp.calculation == "relax"
            || PARAM.inp.calculation == "cell-relax"
            || PARAM.inp.calculation == "md") {
            if (GlobalC::exx_info.info_global.cal_exx)
            {
                XC_Functional::set_xc_first_loop(ucell);
                this->exx_lip = std::unique_ptr<Exx_Lip<T>>(new Exx_Lip<T>(GlobalC::exx_info.info_lip,
                                                                           ucell.symm,
                                                                           &this->kv,
                                                                           this->psi_local,
                                                                           this->kspw_psi,
                                                                           this->pw_wfc,
                                                                           this->pw_rho,
                                                                           this->sf,
                                                                           &ucell,
                                                                           this->pelec));
                // this->exx_lip.init(GlobalC::exx_info.info_lip, cell.symm, &this->kv, this->p_psi_init, this->kspw_psi, this->pw_wfc, this->pw_rho, this->sf, &cell, this->pelec);
            }
}
#endif
    }

    template <typename T>
    void ESolver_KS_LIP<T>::iter_init(UnitCell& ucell, const int istep, const int iter)
    {
        ESolver_KS_PW<T>::iter_init(ucell, istep, iter);
#ifdef __EXX
        if (GlobalC::exx_info.info_global.cal_exx && !GlobalC::exx_info.info_global.separate_loop && this->two_level_step) {
            this->exx_lip->cal_exx();
}
#endif
    }

    template <typename T>
    void ESolver_KS_LIP<T>::hamilt2rho_single(UnitCell& ucell, const int istep, const int iter, const double ethr)
    {
        ModuleBase::TITLE("ESolver_KS_LIP", "hamilt2rho_single");
        ModuleBase::timer::tick("ESolver_KS_LIP", "hamilt2rho_single");

        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        // choose if psi should be diag in subspace
        // be careful that istep start from 0 and iter start from 1
        // if (iter == 1)
        hsolver::DiagoIterAssist<T>::need_subspace = ((istep == 0 || istep == 1) && iter == 1) ? false : true;
        hsolver::DiagoIterAssist<T>::SCF_ITER = iter;
        hsolver::DiagoIterAssist<T>::PW_DIAG_THR = ethr;
        hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX = PARAM.inp.pw_diag_nmax;
        bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;

        hsolver::HSolverLIP<T> hsolver_lip_obj(this->pw_wfc);
        hsolver_lip_obj.solve(this->p_hamilt, this->kspw_psi[0], this->pelec, *this->psi_local, skip_charge,ucell.tpiba,ucell.nat);

        // add exx
#ifdef __EXX
        if (GlobalC::exx_info.info_global.cal_exx)
        {
            this->pelec->set_exx(this->exx_lip->get_exx_energy()); // Peize Lin add 2019-03-09
        }
#endif

        Symmetry_rho srho;
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            srho.begin(is, this->chr, this->pw_rhod, ucell.symm);
        }

        // deband is calculated from "output" charge density calculated
        // in sum_band
        // need 'rho(out)' and 'vr (v_h(in) and v_xc(in))'
        this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);

        ModuleBase::timer::tick("ESolver_KS_LIP", "hamilt2rho_single");
    }

    template <typename T>
    void ESolver_KS_LIP<T>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
    {
        ESolver_KS_PW<T>::iter_finish(ucell, istep, iter, conv_esolver);

#ifdef __EXX
        if (GlobalC::exx_info.info_global.cal_exx && conv_esolver)
        {
            // no separate_loop case
            if (!GlobalC::exx_info.info_global.separate_loop)
            {
                GlobalC::exx_info.info_global.hybrid_step = 1;

                // in no_separate_loop case, scf loop only did twice
                // in first scf loop, exx updated once in beginning,
                // in second scf loop, exx updated every iter

                if (!this->two_level_step)
                {
                    // update exx and redo scf
                    XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);
                    iter = 0;
                    std::cout << " Entering 2nd SCF, where EXX is updated" << std::endl;
                    this->two_level_step++;
                    conv_esolver = false;
                }
            }
            // has separate_loop case
            // exx converged or get max exx steps
            else if (this->two_level_step == GlobalC::exx_info.info_global.hybrid_step
                     || (iter == 1 && this->two_level_step != 0))
            {
                conv_esolver = true;
            }
            else
            {
                // update exx and redo scf
                if (this->two_level_step == 0)
                {
                    XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);
                }

                std::cout << " Updating EXX " << std::flush;
                timeval t_start;
                gettimeofday(&t_start, nullptr);

                this->exx_lip->cal_exx();
                iter = 0;
                this->two_level_step++;

                timeval t_end;
                gettimeofday(&t_end, nullptr);
                std::cout << "and rerun SCF\t" << std::setprecision(3) << std::setiosflags(std::ios::scientific)
                          << (double)(t_end.tv_sec - t_start.tv_sec)
                                 + (double)(t_end.tv_usec - t_start.tv_usec) / 1000000.0
                          << std::defaultfloat << " (s)" << std::endl;
                conv_esolver = false;
            }
        }
#endif
    }

    template <typename T>
    void ESolver_KS_LIP<T>::after_all_runners(UnitCell& ucell)
    {
        ESolver_KS_PW<T>::after_all_runners(ucell);

#ifdef __LCAO
        if (PARAM.inp.out_mat_xc)
        {
            ModuleIO::write_Vxc(PARAM.inp.nspin,
                                PARAM.globalv.nlocal,
                                GlobalV::DRANK,
                                *this->kspw_psi,
                                ucell,
                                this->sf,
                                this->solvent,
                                *this->pw_wfc,
                                *this->pw_rho,
                                *this->pw_rhod,
                                this->locpp.vloc,
                                this->chr,
                                this->kv,
                                this->pelec->wg
#ifdef __EXX
                                ,
                                *this->exx_lip
#endif
            );
        }
#endif
    }
    template class ESolver_KS_LIP<std::complex<float>>;
    template class ESolver_KS_LIP<std::complex<double>>;
    // LIP is not supported on GPU yet.
} // namespace ModuleESolver
