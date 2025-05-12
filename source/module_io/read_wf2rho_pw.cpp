#include "read_wf2rho_pw.h"

#include "read_wfc_pw.h"
#include "module_base/timer.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_parameter/parameter.h"
#include "module_elecstate/kernels/elecstate_op.h"

void ModuleIO::read_wf2rho_pw(const ModulePW::PW_Basis_K* pw_wfc,
                               ModuleSymmetry::Symmetry& symm,
                               const int* ik2iktot,
                               const int nkstot,
                               const std::vector<int>& isk,
                               Charge& chg)
{
    ModuleBase::TITLE("ModuleIO", "read_wf2rho_pw");
    ModuleBase::timer::tick("ModuleIO", "read_wf2rho_pw");

    const int kpar = GlobalV::KPAR;
    const int my_pool = GlobalV::MY_POOL;
    const int my_rank = GlobalV::MY_RANK;
    const int nbands = PARAM.inp.nbands;
    const int nspin = PARAM.inp.nspin;

    const int ng_npol = pw_wfc->npwk_max * PARAM.globalv.npol;
    const int nrxx = pw_wfc->nrxx;
    for (int is = 0; is < nspin; ++is)
    {
        ModuleBase::GlobalFunc::ZEROS(chg.rho[is], nrxx);
    }

    ModuleBase::ComplexMatrix wfc_tmp(nbands, ng_npol);
    std::vector<std::complex<double>> rho_tmp(nrxx);

    // read occupation numbers
    ModuleBase::matrix wg_tmp(nkstot, nbands);
    if (my_rank == 0)
    {
        std::string filename = PARAM.globalv.global_readin_dir + "istate.info";
        std::ifstream ifs(filename);
        std::string useless;
        if (PARAM.inp.nspin == 2)
        {
            const int nkstot_np = nkstot / 2;
            for (int iktot_np = 0; iktot_np < nkstot_np; ++iktot_np)
            {
                ifs >> useless;
                getline(ifs, useless);
                for (int ib = 0; ib < nbands; ++ib)
                {
                    ifs >> useless >> useless >> wg_tmp(iktot_np, ib) >> useless >> wg_tmp(iktot_np + nkstot_np, ib);
                }
            }
        }
        else
        {
            for (int ik_tot = 0; ik_tot < nkstot; ++ik_tot)
            {
                ifs >> useless;
                getline(ifs, useless);
                for (int ib = 0; ib < nbands; ++ib)
                {
                    ifs >> useless >> useless >> wg_tmp(ik_tot, ib);
                }
            }
        }
    }

#ifdef __MPI
    MPI_Bcast(wg_tmp.c, nkstot * nbands, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    for (int ik = 0; ik < pw_wfc->nks; ++ik)
    {
        int is = 0;
        if (nspin == 2)
        {
            is = isk[ik];
        }
        std::stringstream filename;
        const int ikstot = ik2iktot[ik];
        filename << PARAM.globalv.global_readin_dir << "WAVEFUNC" << ikstot + 1 << ".dat";
        ModuleIO::read_wfc_pw(filename.str(), pw_wfc, ik, ikstot, nkstot, wfc_tmp);
        if (PARAM.inp.nspin == 4)
        {
            std::vector<std::complex<double>> rho_tmp2(nrxx);
            for (int ib = 0; ib < nbands; ++ib)
            {
                const std::complex<double>* wfc_ib = wfc_tmp.c + ib * ng_npol;
                const std::complex<double>* wfc_ib2 = wfc_tmp.c + ib * ng_npol + ng_npol / 2;
                pw_wfc->recip2real(wfc_ib, rho_tmp.data(), ik);
                pw_wfc->recip2real(wfc_ib2, rho_tmp2.data(), ik);
                const double w1 = wg_tmp(ikstot, ib) / pw_wfc->omega;

                if (w1 != 0.0)
                {
                    base_device::DEVICE_CPU* ctx = nullptr;
                    elecstate::elecstate_pw_op<double, base_device::DEVICE_CPU>()(ctx,
                                                                                  PARAM.globalv.domag,
                                                                                  PARAM.globalv.domag_z,
                                                                                  nrxx,
                                                                                  w1,
                                                                                  chg.rho,
                                                                                  rho_tmp.data(),
                                                                                  rho_tmp2.data());
                }
            }
        }
        else
        {
            for (int ib = 0; ib < nbands; ++ib)
            {
                const std::complex<double>* wfc_ib = wfc_tmp.c + ib * ng_npol;
                pw_wfc->recip2real(wfc_ib, rho_tmp.data(), ik);

                const double w1 = wg_tmp(ikstot, ib) / pw_wfc->omega;

                if (w1 != 0.0)
                {
                    base_device::DEVICE_CPU* ctx = nullptr;
                    elecstate::elecstate_pw_op<double, base_device::DEVICE_CPU>()(ctx, is, nrxx, w1, chg.rho, rho_tmp.data());
                }
            }
        }
    }

#ifdef __MPI
    chg.init_chgmpi();
    for (int is = 0; is < nspin; ++is)
    {
        chg.reduce_diff_pools(chg.rho[is]);
    }
#endif

    // Since rho is calculated by psi^2, it is not symmetric. We need to rearrange it. 
    Symmetry_rho srho;
    for (int is = 0; is < nspin; is++)
    {
        srho.begin(is, chg, chg.rhopw, symm);
    }

    ModuleBase::timer::tick("ModuleIO", "read_wf2rho_pw");
}
