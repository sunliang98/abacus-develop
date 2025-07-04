#include "to_wannier90_lcao_in_pw.h"

#include "module_parameter/parameter.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_base/math_integral.h"
#include "source_base/math_polyint.h"
#include "source_base/math_sphbes.h"
#include "source_base/math_ylmreal.h"
#include "source_base/parallel_reduce.h"
#include "binstream.h"

#include "source_psi/psi_initializer_nao.h"
#ifdef __LCAO
toWannier90_LCAO_IN_PW::toWannier90_LCAO_IN_PW(
    const bool &out_wannier_mmn, 
    const bool &out_wannier_amn, 
    const bool &out_wannier_unk, 
    const bool &out_wannier_eig,
    const bool &out_wannier_wvfn_formatted, 
    const std::string &nnkpfile,
    const std::string &wannier_spin
):toWannier90_PW(out_wannier_mmn, out_wannier_amn, out_wannier_unk, out_wannier_eig, out_wannier_wvfn_formatted, nnkpfile, wannier_spin)
{
}

toWannier90_LCAO_IN_PW::~toWannier90_LCAO_IN_PW()
{
    delete psi_initer_;
    delete psi;   
}

void toWannier90_LCAO_IN_PW::calculate(
    UnitCell& ucell,
    const ModuleBase::matrix& ekb,
    const ModulePW::PW_Basis_K* wfcpw,
    const ModulePW::PW_Basis_Big* bigpw,
    const Structure_Factor& sf,
    const K_Vectors& kv,
    const psi::Psi<std::complex<double>>* psi,
    const Parallel_Orbitals *pv
)
{
    this->ParaV = pv;

    Structure_Factor* sf_ptr = const_cast<Structure_Factor*>(&sf);
    ModulePW::PW_Basis_K* wfcpw_ptr = const_cast<ModulePW::PW_Basis_K*>(wfcpw);
    delete this->psi_initer_;
    this->psi_initer_ = new psi_initializer_nao<std::complex<double>>();
    this->psi_initer_->initialize(sf_ptr, wfcpw_ptr, &ucell, &kv, 1, nullptr, GlobalV::MY_RANK);
    this->psi_initer_->tabulate();
    delete this->psi;
    const int nks_psi = (PARAM.inp.calculation == "nscf" && PARAM.inp.mem_saver == 1)? 1 : wfcpw->nks;
    const int nks_psig = (PARAM.inp.basis_type == "pw")? 1 : nks_psi;
    const int nbands_actual = this->psi_initer_->nbands_start();
    this->psi = new psi::Psi<std::complex<double>, base_device::DEVICE_CPU>(nks_psig, 
                                                                            nbands_actual, 
                                                                            wfcpw->npwk_max*PARAM.globalv.npol, 
                                                                            kv.ngk,
                                                                            true);
    read_nnkp(ucell,kv);

    if (PARAM.inp.nspin == 2)
    {
        if (wannier_spin == "up")
        {
            start_k_index = 0;
        }
        else if (wannier_spin == "down")
        {
            start_k_index = num_kpts / 2;
        }
        else
        {
            ModuleBase::WARNING_QUIT("toWannier90::calculate", "Error wannier_spin set,is not \"up\" or \"down\" ");
        }
    }

    psi::Psi<std::complex<double>> *unk_inLcao = get_unk_from_lcao(ucell,*psi, wfcpw, sf, kv);

    if (out_wannier_eig)
    {
        out_eig(ekb);
    }

#ifdef __MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // To calculate the Mmn and Amn files, cal_band_index needs to be modified, 
    // because the wave function unk_inLcao only stores the energy bands that need to be calculated.
    for (int ib = 0; ib < num_bands; ib++)
    {
        cal_band_index[ib] = ib;
    }

    if (out_wannier_mmn)
    {
        cal_Mmn(*unk_inLcao, wfcpw);
    }

    if (out_wannier_amn)
    {
        cal_Amn(*unk_inLcao, wfcpw);
    }

    if (out_wannier_unk)
    {
        out_unk(*unk_inLcao, wfcpw, bigpw);
    }

    delete unk_inLcao;
}

psi::Psi<std::complex<double>>* toWannier90_LCAO_IN_PW::get_unk_from_lcao(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>& psi_in, 
    const ModulePW::PW_Basis_K* wfcpw,
    const Structure_Factor& sf,
    const K_Vectors& kv
)
{
    // init
    int npwx = wfcpw->npwk_max;
    psi::Psi<std::complex<double>> *unk_inLcao = new psi::Psi<std::complex<double>>(num_kpts, 
                                                                                    num_bands, 
                                                                                    npwx*PARAM.globalv.npol, 
                                                                                    kv.ngk,
                                                                                    true);
    unk_inLcao->zero_out();

    // Orbital projection to plane wave
    ModuleBase::realArray table_local(ucell.ntype, ucell.nmax_total, PARAM.globalv.nqx);

    for (int ik = 0; ik < num_kpts; ik++)
    {
        int npw = kv.ngk[ik];
        ModuleBase::ComplexMatrix orbital_in_G(PARAM.globalv.nlocal, npwx*PARAM.globalv.npol);
        // Wavefunc_in_pw::produce_local_basis_in_pw(ik, wfcpw, sf, orbital_in_G, table_local);
        //produce_local_basis_in_pw(ik, wfcpw, sf, orbital_in_G, table_local);
        nao_G_expansion(ik, wfcpw, orbital_in_G);

        ModuleBase::ComplexMatrix lcao_wfc_global;
        get_lcao_wfc_global_ik(ik, psi_in, lcao_wfc_global);

        if (PARAM.inp.nspin != 4)
        {
            for (int ib = 0; ib < num_bands; ib++)
            {
                for (int ig = 0; ig < npw; ig++)
                {
                    for (int iw = 0; iw < PARAM.globalv.nlocal; iw++)
                    {
                        unk_inLcao[0](ik, ib, ig) +=  lcao_wfc_global(ib, iw) * orbital_in_G(iw, ig);
                    }
                }

                std::complex<double> anorm(0.0, 0.0);
                for (int ig = 0; ig < npw; ig++)
                {
                    anorm = anorm + conj(unk_inLcao[0](ik, ib, ig)) * unk_inLcao[0](ik, ib, ig);
                }

#ifdef __MPI
                Parallel_Reduce::reduce_all(anorm);
#endif

                for (int ig = 0; ig < npw; ig++)
                {
                    unk_inLcao[0](ik, ib, ig) = unk_inLcao[0](ik, ib, ig) / sqrt(anorm);
                }
            }
        }
        else
        {
            for (int ib = 0; ib < num_bands; ib++)
            {
                // for (int ig = 0; ig < npwx*PARAM.globalv.npol; ig++)
                // {
                //     for (int iw = 0; iw < PARAM.globalv.nlocal; iw++)
                //     {
                //         unk_inLcao[0](ik, ib, ig) +=  lcao_wfc_global(ib, iw) * orbital_in_G(iw, ig);
                //     }
                // }

                for (int ig = 0; ig < npw; ig++)
                {
                    int basis_num = PARAM.globalv.nlocal / 2;
                    for (int iw = 0; iw < basis_num; iw++)
                    {
                        unk_inLcao[0](ik, ib, ig) +=  lcao_wfc_global(ib, 2*iw) * orbital_in_G(iw, ig);
                        unk_inLcao[0](ik, ib, ig+npwx) +=  lcao_wfc_global(ib, 2*iw+1) * orbital_in_G(iw, ig);
                    }
                }

                std::complex<double> anorm(0.0, 0.0);
                for (int ig = 0; ig < npw; ig++)
                {
                    anorm = anorm + conj(unk_inLcao[0](ik, ib, ig)) * unk_inLcao[0](ik, ib, ig) + conj(unk_inLcao[0](ik, ib, ig+npwx)) * unk_inLcao[0](ik, ib, ig+npwx);
                }

#ifdef __MPI
                Parallel_Reduce::reduce_all(anorm);
#endif

                for (int ig = 0; ig < npw; ig++)
                {
                    unk_inLcao[0](ik, ib, ig) = unk_inLcao[0](ik, ib, ig) / sqrt(anorm);
                    unk_inLcao[0](ik, ib, ig+npwx) = unk_inLcao[0](ik, ib, ig+npwx) / sqrt(anorm);
                }
            }
        }

    }

#ifdef __MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return unk_inLcao;
}

void toWannier90_LCAO_IN_PW::nao_G_expansion(
    const int& ik,
    const ModulePW::PW_Basis_K* wfcpw,
    ModuleBase::ComplexMatrix& psi
)
{
    int npwx = wfcpw->npwk_max;
    this->psi->fix_k(ik);
    this->psi_initer_->init_psig(this->psi->get_pointer(), ik);
    int nbands = PARAM.globalv.nlocal;
    int nbasis = npwx*PARAM.globalv.npol;
    for (int ib = 0; ib < nbands; ib++)
    {
        for (int ig = 0; ig < nbasis; ig++)
        {
            psi(ib, ig) = this->psi->operator()(ib, ig);
        }
    }
}

void toWannier90_LCAO_IN_PW::get_lcao_wfc_global_ik(
    const int ik, 
    const psi::Psi<std::complex<double>>& psi_in, 
    ModuleBase::ComplexMatrix &lcao_wfc_global
)
{
    lcao_wfc_global.create(num_bands, PARAM.globalv.nlocal);

    int count_b = -1;
    int row = this->ParaV->get_row_size();
    int global_row_index = 0;
    for (int ib = 0; ib < PARAM.inp.nbands; ib++)
    {
        if (exclude_bands.count(ib)) { continue;
}
        count_b++;

        int ic = this->ParaV->global2local_col(ib);

        if (ic >= 0)
        {
            for (int ir = 0; ir < row; ir++)
            {
                global_row_index = this->ParaV->local2global_row(ir);
                lcao_wfc_global(count_b, global_row_index) = psi_in(ik, ic, ir);
            }
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_all(lcao_wfc_global.c, lcao_wfc_global.size);
#endif

}
#endif
