#include "esolver_ks_lcao_tddft.h"

#include "module_io/cal_r_overlap_R.h"
#include "module_io/dipole_io.h"
#include "module_io/dm_io.h"
#include "module_io/rho_io.h"
#include "module_io/td_current_io.h"
#include "module_io/write_HS.h"
#include "module_io/write_HS_R.h"

//--------------temporary----------------------------
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_base/scalapack_connector.h"
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_lcao/module_tddft/evolve_elec.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/print_info.h"

//-----HSolver ElecState Hamilt--------
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/elecstate_lcao_tddft.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hsolver/hsolver_lcao.h"
#include "module_psi/psi.h"

//-----force& stress-------------------
#include "module_hamilt_lcao/hamilt_lcaodft/FORCE_STRESS.h"

//---------------------------------------------------

namespace ModuleESolver
{

ESolver_KS_LCAO_TDDFT::ESolver_KS_LCAO_TDDFT()
{
    classname = "ESolver_KS_LCAO_TDDFT";
    basisname = "LCAO";
}

ESolver_KS_LCAO_TDDFT::~ESolver_KS_LCAO_TDDFT()
{
    // this->orb_con.clear_after_ions(GlobalC::UOT, GlobalC::ORB, GlobalV::deepks_setorb, GlobalC::ucell.infoNL.nproj);
    delete psi_laststep;
    if (Hk_laststep != nullptr)
    {
        for (int ik = 0; ik < kv.nks; ++ik)
        {
            delete[] Hk_laststep[ik];
        }
        delete[] Hk_laststep;
    }
    if (Sk_laststep != nullptr)
    {
        for (int ik = 0; ik < kv.nks; ++ik)
        {
            delete[] Sk_laststep[ik];
        }
        delete[] Sk_laststep;
    }
}

void ESolver_KS_LCAO_TDDFT::before_all_runners(Input& inp, UnitCell& ucell)
{
    ESolver_KS::before_all_runners(inp, ucell);

    // Initialize the FFT.
    // this function belongs to cell LOOP

    // output is GlobalC::ppcell.vloc 3D local pseudopotentials
    // without structure factors
    // this function belongs to cell LOOP
    GlobalC::ppcell.init_vloc(GlobalC::ppcell.vloc, pw_rho);

    if (this->pelec == nullptr)
    {
        this->pelec = new elecstate::ElecStateLCAO_TDDFT(&(this->chr),
                                                         &(kv),
                                                         kv.nks,
                                                         &(this->LOC),
                                                         &(this->GK), // mohan add 2024-04-01
                                                         &(this->LOWF),
                                                         this->pw_rho,
                                                         pw_big);
    }

    //------------------init Basis_lcao----------------------
    // Init Basis should be put outside of Ensolver.
    // * reading the localized orbitals/projectors
    // * construct the interpolation tables.
    this->init_basis_lcao(this->orb_con, inp, ucell);
    //------------------init Basis_lcao----------------------

    //------------------init Hamilt_lcao----------------------
    // * allocate H and S matrices according to computational resources
    // * set the 'trace' between local H/S and global H/S
    this->LM.divide_HS_in_frag(GlobalV::GAMMA_ONLY_LOCAL, orb_con.ParaV, kv.nks);
    //------------------init Hamilt_lcao----------------------

    // pass Hamilt-pointer to Operator
    this->gen_h.LM = &this->LM;
    // pass basis-pointer to EState and Psi
    this->LOC.ParaV = this->LOWF.ParaV = this->LM.ParaV;

    // init DensityMatrix
    dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)
        ->init_DM(&kv, this->LM.ParaV, GlobalV::NSPIN);

    // init Psi, HSolver, ElecState, Hamilt
    if (this->phsol == nullptr)
    {
        this->phsol = new hsolver::HSolverLCAO<std::complex<double>>(this->LOWF.ParaV);
        this->phsol->method = GlobalV::KS_SOLVER;
    }

    // Inititlize the charge density.
    this->pelec->charge->allocate(GlobalV::NSPIN);
    this->pelec->omega = GlobalC::ucell.omega;

    // Initializee the potential.
    this->pelec->pot = new elecstate::Potential(pw_rhod,
                                                pw_rho,
                                                &GlobalC::ucell,
                                                &(GlobalC::ppcell.vloc),
                                                &(sf),
                                                &(pelec->f_en.etxc),
                                                &(pelec->f_en.vtxc));

    this->pelec_td = dynamic_cast<elecstate::ElecStateLCAO_TDDFT*>(this->pelec);
}

void ESolver_KS_LCAO_TDDFT::hamilt2density(const int istep, const int iter, const double ethr)
{
    pelec->charge->save_rho_before_sum_band();

    if (wf.init_wfc == "file")
    {
        if (istep >= 1)
        {
            module_tddft::Evolve_elec::solve_psi(istep,
                                                 GlobalV::NBANDS,
                                                 GlobalV::NLOCAL,
                                                 this->p_hamilt,
                                                 this->LOWF,
                                                 this->psi,
                                                 this->psi_laststep,
                                                 this->Hk_laststep,
                                                 this->Sk_laststep,
                                                 this->pelec_td->ekb,
                                                 td_htype,
                                                 INPUT.propagator,
                                                 kv.nks);
            this->pelec_td->psiToRho_td(this->psi[0]);
        }
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    else if (istep >= 2)
    {
        module_tddft::Evolve_elec::solve_psi(istep,
                                             GlobalV::NBANDS,
                                             GlobalV::NLOCAL,
                                             this->p_hamilt,
                                             this->LOWF,
                                             this->psi,
                                             this->psi_laststep,
                                             this->Hk_laststep,
                                             this->Sk_laststep,
                                             this->pelec_td->ekb,
                                             td_htype,
                                             INPUT.propagator,
                                             kv.nks);
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    // using HSolverLCAO<std::complex<double>>::solve()
    else if (this->phsol != nullptr)
    {
        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        if (this->psi != nullptr)
        {
            this->phsol->solve(this->p_hamilt, this->psi[0], this->pelec_td, GlobalV::KS_SOLVER);
        }
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_LCAO", "HSolver has not been initialed!");
    }

    // print occupation of each band
    if (iter == 1 && istep <= 2)
    {
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << std::endl;
        GlobalV::ofs_running << "occupation : " << std::endl;
        GlobalV::ofs_running << "ik  iband     occ " << std::endl;
        GlobalV::ofs_running << std::setprecision(6);
        GlobalV::ofs_running << std::setiosflags(std::ios::showpoint);
        for (int ik = 0; ik < kv.nks; ik++)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ib++)
            {
                std::setprecision(6);
                GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      " << this->pelec_td->wg(ik, ib)
                                     << std::endl;
            }
        }
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << std::endl;
    }

    for (int ik = 0; ik < kv.nks; ++ik)
    {
        this->pelec_td->print_band(ik, INPUT.printe, iter);
    }

    // using new charge density.
    this->pelec->cal_energies(1);

    // symmetrize the charge density only for ground state
    if (istep <= 1)
    {
        Symmetry_rho srho;
        for (int is = 0; is < GlobalV::NSPIN; is++)
        {
            srho.begin(is, *(pelec->charge), pw_rho, GlobalC::Pgrid, GlobalC::ucell.symm);
        }
    }

    // (6) compute magnetization, only for spin==2
    GlobalC::ucell.magnet.compute_magnetization(this->pelec->charge->nrxx,
                                                this->pelec->charge->nxyz,
                                                this->pelec->charge->rho,
                                                pelec->nelec_spin.data());

    // (7) calculate delta energy
    this->pelec->f_en.deband = this->pelec->cal_delta_eband();
}

void ESolver_KS_LCAO_TDDFT::update_pot(const int istep, const int iter)
{
    // print Hamiltonian and Overlap matrix
    if (this->conv_elec)
    {
        if (!GlobalV::GAMMA_ONLY_LOCAL)
        {
            this->GK.renew(true);
        }
        for (int ik = 0; ik < kv.nks; ++ik)
        {
            if (hsolver::HSolverLCAO<std::complex<double>>::out_mat_hs[0])
            {
                this->p_hamilt->updateHk(ik);
            }
            bool bit = false; // LiuXh, 2017-03-21
            // if set bit = true, there would be error in soc-multi-core calculation, noted by zhengdy-soc
            if (this->psi != nullptr && (istep % GlobalV::out_interval == 0))
            {
                hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
                this->p_hamilt->matrix(h_mat, s_mat);
                if (hsolver::HSolverLCAO<std::complex<double>>::out_mat_hs[0])
                {
                    ModuleIO::save_mat(istep,
                                       h_mat.p,
                                       GlobalV::NLOCAL,
                                       bit,
                                       hsolver::HSolverLCAO<std::complex<double>>::out_mat_hs[1],
                                       1,
                                       GlobalV::out_app_flag,
                                       "H",
                                       "data-" + std::to_string(ik),
                                       *this->LOWF.ParaV,
                                       GlobalV::DRANK);

                    ModuleIO::save_mat(istep,
                                       s_mat.p,
                                       GlobalV::NLOCAL,
                                       bit,
                                       hsolver::HSolverLCAO<std::complex<double>>::out_mat_hs[1],
                                       1,
                                       GlobalV::out_app_flag,
                                       "S",
                                       "data-" + std::to_string(ik),
                                       *this->LOWF.ParaV,
                                       GlobalV::DRANK);
                }
            }
        }
    }

    if (this->conv_elec)
    {
        if (elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_lcao)
        {
            elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_flag
                = elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_lcao;
        }
        for (int ik = 0; ik < kv.nks; ik++)
        {
            if (istep % GlobalV::out_interval == 0)
            {
                this->psi[0].fix_k(ik);
                this->pelec->print_psi(this->psi[0], istep);
            }
        }
        elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_flag = 0;
    }

    // Calculate new potential according to new Charge Density
    if (!this->conv_elec)
    {
        if (GlobalV::NSPIN == 4)
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

    // store wfc and Hk laststep
    if (istep >= (wf.init_wfc == "file" ? 0 : 1) && this->conv_elec)
    {
        if (this->psi_laststep == nullptr)
        {
#ifdef __MPI
            this->psi_laststep = new psi::Psi<std::complex<double>>(kv.nks,
                                                                    this->LOWF.ParaV->ncol_bands,
                                                                    this->LOWF.ParaV->nrow,
                                                                    nullptr);
#else
            this->psi_laststep = new psi::Psi<std::complex<double>>(kv.nks, GlobalV::NBANDS, GlobalV::NLOCAL, nullptr);
#endif
        }

        if (td_htype == 1)
        {
            if (this->Hk_laststep == nullptr)
            {
                this->Hk_laststep = new std::complex<double>*[kv.nks];
                for (int ik = 0; ik < kv.nks; ++ik)
                {
                    this->Hk_laststep[ik] = new std::complex<double>[this->LOC.ParaV->nloc];
                    ModuleBase::GlobalFunc::ZEROS(Hk_laststep[ik], this->LOC.ParaV->nloc);
                }
            }
            if (this->Sk_laststep == nullptr)
            {
                this->Sk_laststep = new std::complex<double>*[kv.nks];
                for (int ik = 0; ik < kv.nks; ++ik)
                {
                    this->Sk_laststep[ik] = new std::complex<double>[this->LOC.ParaV->nloc];
                    ModuleBase::GlobalFunc::ZEROS(Sk_laststep[ik], this->LOC.ParaV->nloc);
                }
            }
        }

        for (int ik = 0; ik < kv.nks; ++ik)
        {
            this->psi->fix_k(ik);
            this->psi_laststep->fix_k(ik);
            int size0 = psi->get_nbands() * psi->get_nbasis();
            for (int index = 0; index < size0; ++index)
            {
                psi_laststep[0].get_pointer()[index] = psi[0].get_pointer()[index];
            }

            // store Hamiltonian
            if (td_htype == 1)
            {
                this->p_hamilt->updateHk(ik);
                hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
                this->p_hamilt->matrix(h_mat, s_mat);
                BlasConnector::copy(this->LOC.ParaV->nloc, h_mat.p, 1, Hk_laststep[ik], 1);
                BlasConnector::copy(this->LOC.ParaV->nloc, s_mat.p, 1, Sk_laststep[ik], 1);
            }
        }

        // calculate energy density matrix for tddft
        if (istep >= (wf.init_wfc == "file" ? 0 : 2) && module_tddft::Evolve_elec::td_edm == 0)
        {
            this->cal_edm_tddft();
        }
    }

    // print "eigen value" for tddft
    if (this->conv_elec)
    {
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << std::endl;
        GlobalV::ofs_running << "Eii : " << std::endl;
        GlobalV::ofs_running << "ik  iband    Eii (eV)" << std::endl;
        GlobalV::ofs_running << std::setprecision(6);
        GlobalV::ofs_running << std::setiosflags(std::ios::showpoint);

        for (int ik = 0; ik < kv.nks; ik++)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ib++)
            {
                GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      "
                                     << this->pelec_td->ekb(ik, ib) * ModuleBase::Ry_to_eV << std::endl;
            }
        }
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running
            << "------------------------------------------------------------------------------------------------"
            << std::endl;
    }
}

void ESolver_KS_LCAO_TDDFT::after_scf(const int istep)
{
    for (int is = 0; is < GlobalV::NSPIN; is++)
    {
        if (module_tddft::Evolve_elec::out_dipole == 1)
        {
            std::stringstream ss_dipole;
            ss_dipole << GlobalV::global_out_dir << "SPIN" << is + 1 << "_DIPOLE";
            ModuleIO::write_dipole(pelec->charge->rho_save[is], pelec->charge->rhopw, is, istep, ss_dipole.str());
        }
    }
    if (module_tddft::Evolve_elec::out_current == 1)
    {
        elecstate::DensityMatrix<std::complex<double>, double>* tmp_DM
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM();
        ModuleIO::write_current(istep,
                                this->psi,
                                pelec,
                                kv,
                                tmp_DM->get_paraV_pointer(),
                                this->RA,
                                this->LM,     // mohan add 2024-04-02
                                this->gen_h); // mohan add 2024-02
    }
    ESolver_KS_LCAO<std::complex<double>, double>::after_scf(istep);
}

// use the original formula (Hamiltonian matrix) to calculate energy density matrix
void ESolver_KS_LCAO_TDDFT::cal_edm_tddft(void)
{
    // mohan add 2024-03-27
    const int nlocal = GlobalV::NLOCAL;
    assert(nlocal >= 0);

    // this->LOC.edm_k_tddft.resize(kv.nks);
    dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->EDMK.resize(kv.nks);
    for (int ik = 0; ik < kv.nks; ++ik)
    {
        std::complex<double>* tmp_dmk
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_DMK_pointer(ik);

        ModuleBase::ComplexMatrix& tmp_edmk
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->EDMK[ik];

        const Parallel_Orbitals* tmp_pv
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->get_paraV_pointer();

#ifdef __MPI

        // mohan add 2024-03-27
        //! be careful, the type of nloc is 'long'
        //! whether the long type is safe, needs more discussion
        const long nloc = this->LOC.ParaV->nloc;

        // this->LOC.edm_k_tddft[ik].create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        tmp_edmk.create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        complex<double>* Htmp = new complex<double>[nloc];
        complex<double>* Sinv = new complex<double>[nloc];
        complex<double>* tmp1 = new complex<double>[nloc];
        complex<double>* tmp2 = new complex<double>[nloc];
        complex<double>* tmp3 = new complex<double>[nloc];
        complex<double>* tmp4 = new complex<double>[nloc];

        ModuleBase::GlobalFunc::ZEROS(Htmp, nloc);
        ModuleBase::GlobalFunc::ZEROS(Sinv, nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp1, nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp2, nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp3, nloc);
        ModuleBase::GlobalFunc::ZEROS(tmp4, nloc);

        const int inc = 1;
        int nrow = this->LOC.ParaV->nrow;
        int ncol = this->LOC.ParaV->ncol;

        hamilt::MatrixBlock<complex<double>> h_mat;
        hamilt::MatrixBlock<complex<double>> s_mat;

        p_hamilt->matrix(h_mat, s_mat);
        zcopy_(&nloc, h_mat.p, &inc, Htmp, &inc);
        zcopy_(&nloc, s_mat.p, &inc, Sinv, &inc);

        int* ipiv = new int[nloc];
        int info;
        const int one_int = 1;
        pzgetrf_(&nlocal, &nlocal, Sinv, &one_int, &one_int, this->LOC.ParaV->desc, ipiv, &info);

        int LWORK = -1, liWORK = -1;
        std::vector<std::complex<double>> WORK(1, 0);
        std::vector<int> iWORK(1, 0);

        pzgetri_(&nlocal,
                 Sinv,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 ipiv,
                 WORK.data(),
                 &LWORK,
                 iWORK.data(),
                 &liWORK,
                 &info);

        LWORK = WORK[0].real();
        WORK.resize(LWORK, 0);
        liWORK = iWORK[0];
        iWORK.resize(liWORK, 0);

        pzgetri_(&nlocal,
                 Sinv,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 ipiv,
                 WORK.data(),
                 &LWORK,
                 iWORK.data(),
                 &liWORK,
                 &info);

        const char N_char = 'N', T_char = 'T';
        const complex<double> one_float = {1.0, 0.0}, zero_float = {0.0, 0.0};
        const complex<double> half_float = {0.5, 0.0};
        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                tmp_dmk,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Htmp,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp1,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                tmp1,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Sinv,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp2,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                Sinv,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                Htmp,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp3,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                tmp3,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                tmp_dmk,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc,
                &zero_float,
                tmp4,
                &one_int,
                &one_int,
                this->LOC.ParaV->desc);

        pzgeadd_(&N_char,
                 &nlocal,
                 &nlocal,
                 &half_float,
                 tmp2,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc,
                 &half_float,
                 tmp4,
                 &one_int,
                 &one_int,
                 this->LOC.ParaV->desc);
        zcopy_(&nloc, tmp4, &inc, tmp_edmk.c, &inc);
        // zcopy_(&nloc, tmp4, &inc, this->LOC.edm_k_tddft[ik].c, &inc);
        delete[] Htmp;
        delete[] Sinv;
        delete[] tmp1;
        delete[] tmp2;
        delete[] tmp3;
        delete[] tmp4;
        delete[] ipiv;
#else
        // for serial version
        // this->LOC.edm_k_tddft[ik].create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        tmp_edmk.create(this->LOC.ParaV->ncol, this->LOC.ParaV->nrow);
        ModuleBase::ComplexMatrix Sinv(nlocal, nlocal);
        ModuleBase::ComplexMatrix Htmp(nlocal, nlocal);

        hamilt::MatrixBlock<complex<double>> h_mat;
        hamilt::MatrixBlock<complex<double>> s_mat;

        p_hamilt->matrix(h_mat, s_mat);
        // cout<<"hmat "<<h_mat.p[0]<<endl;
        for (int i = 0; i < nlocal; i++)
        {
            for (int j = 0; j < nlocal; j++)
            {
                Htmp(i, j) = h_mat.p[i * nlocal + j];
                Sinv(i, j) = s_mat.p[i * nlocal + j];
            }
        }
        int INFO = 0;

        int lwork = 3 * nlocal - 1; // tmp
        std::complex<double>* work = new std::complex<double>[lwork];
        ModuleBase::GlobalFunc::ZEROS(work, lwork);

        int IPIV[nlocal];

        LapackConnector::zgetrf(nlocal, nlocal, Sinv, nlocal, IPIV, &INFO);
        LapackConnector::zgetri(nlocal, Sinv, nlocal, IPIV, work, lwork, &INFO);
        // I just use ModuleBase::ComplexMatrix temporarily, and will change it to complex<double>*
        ModuleBase::ComplexMatrix tmp_dmk_base(nlocal, nlocal);
        for (int i = 0; i < nlocal; i++)
        {
            for (int j = 0; j < nlocal; j++)
            {
                tmp_dmk_base(i, j) = tmp_dmk[i * nlocal + j];
            }
        }
        tmp_edmk = 0.5 * (Sinv * Htmp * tmp_dmk_base + tmp_dmk_base * Htmp * Sinv);
        delete[] work;
#endif
    }
    return;
}
} // namespace ModuleESolver
