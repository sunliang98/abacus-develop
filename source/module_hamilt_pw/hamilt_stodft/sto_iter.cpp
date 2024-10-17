#include "sto_iter.h"

#include "module_parameter/parameter.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_quit.h"
#include "module_base/tool_title.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

Stochastic_Iter::Stochastic_Iter()
{
    change = false;
    mu0 = 0;
    method = 2;
}

Stochastic_Iter::~Stochastic_Iter()
{
}

void Stochastic_Iter::init(K_Vectors* pkv_in,
                           ModulePW::PW_Basis_K* wfc_basis,
                           Stochastic_WF& stowf,
                           StoChe<double>& stoche)
{
    p_che = stoche.p_che;
    spolyv = stoche.spolyv;
    nchip = stowf.nchip;
    targetne = PARAM.inp.nelec;
    this->pkv = pkv_in;
    this->method = stoche.method_sto;

    this->stohchi.init(wfc_basis, pkv, &stoche.emin_sto, &stoche.emax_sto);
    this->stofunc.set_E_range(&stoche.emin_sto, &stoche.emax_sto);
}

void Stochastic_Iter::orthog(const int& ik, psi::Psi<std::complex<double>>& psi, Stochastic_WF& stowf)
{
    ModuleBase::TITLE("Stochastic_Iter", "orthog");
    // orthogonal part
    if (PARAM.inp.nbands > 0)
    {
        const int nchipk = stowf.nchip[ik];
        const int npw = psi.get_current_nbas();
        const int npwx = psi.get_nbasis();
        stowf.chi0->fix_k(ik);
        stowf.chiortho->fix_k(ik);
        std::complex<double>*wfgin = stowf.chi0->get_pointer(), *wfgout = stowf.chiortho->get_pointer();
        for (int ig = 0; ig < npwx * nchipk; ++ig)
        {
            wfgout[ig] = wfgin[ig];
        }

        // orthogonal part
        std::complex<double>* sum = new std::complex<double>[PARAM.inp.nbands * nchipk];
        char transC = 'C';
        char transN = 'N';

        // sum(b<NBANDS, a<nchi) = < psi_b | chi_a >
        zgemm_(&transC,
               &transN,
               &PARAM.inp.nbands,
               &nchipk,
               &npw,
               &ModuleBase::ONE,
               &psi(ik, 0, 0),
               &npwx,
               wfgout,
               &npwx,
               &ModuleBase::ZERO,
               sum,
               &PARAM.inp.nbands);
        Parallel_Reduce::reduce_pool(sum, PARAM.inp.nbands * nchipk);

        // psi -= psi * sum
        zgemm_(&transN,
               &transN,
               &npw,
               &nchipk,
               &PARAM.inp.nbands,
               &ModuleBase::NEG_ONE,
               &psi(ik, 0, 0),
               &npwx,
               sum,
               &PARAM.inp.nbands,
               &ModuleBase::ONE,
               wfgout,
               &npwx);
        delete[] sum;
    }
}

void Stochastic_Iter::checkemm(const int& ik, const int istep, const int iter, Stochastic_WF& stowf)
{
    ModuleBase::TITLE("Stochastic_Iter", "checkemm");
    // iter = 1,2,...   istep = 0,1,2,...
    //  if( istep%PARAM.inp.initsto_freq != 0 )    return;
    const int npw = stowf.ngk[ik];
    const int nks = stowf.nks;
    if (istep == 0)
    {
        if (iter > 5)
        {
            return;
        }
    }
    else
    {
        if (iter > 1)
        {
            return;
        }
    }

    const int norder = p_che->norder;
    std::complex<double>* pchi;
    int ntest = 1;

    if (nchip[ik] < ntest)
    {
        ntest = nchip[ik];
    }

    for (int ichi = 0; ichi < ntest; ++ichi)
    {
        if (PARAM.inp.nbands > 0)
        {
            pchi = &stowf.chiortho->operator()(ik, ichi, 0);
        }
        else
        {
            pchi = &stowf.chi0->operator()(ik, ichi, 0);
        }
        while (true)
        {
            bool converge;
            auto hchi_norm = std::bind(&Stochastic_hchi::hchi_norm,
                                       &stohchi,
                                       std::placeholders::_1,
                                       std::placeholders::_2,
                                       std::placeholders::_3);
            converge = p_che->checkconverge(hchi_norm, pchi, npw, *stohchi.Emax, *stohchi.Emin, 5.0);

            if (!converge)
            {
                change = true;
            }
            else
            {
                break;
            }
        }
    }
    if (ik == nks - 1)
    {
#ifdef __MPI
        MPI_Allreduce(MPI_IN_PLACE, stohchi.Emax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, stohchi.Emin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
#endif
        if (change)
        {
            GlobalV::ofs_running << "New Emax Ry" << *stohchi.Emax << " ; new Emin " << *stohchi.Emin << " Ry"
                                 << std::endl;
        }
        change = false;
    }
}

void Stochastic_Iter::check_precision(const double ref, const double thr, const std::string info)
{
    //==============================
    // precision check
    //==============================
    double error = 0;
    if (this->method == 1)
    {
        error = p_che->coef_real[p_che->norder - 1] * spolyv[p_che->norder - 1];
    }
    else
    {
        const int norder = p_che->norder;
        double last_coef = p_che->coef_real[norder - 1];
        double last_spolyv = spolyv[norder * norder - 1];
        error = last_coef
                * (BlasConnector::dot(norder, p_che->coef_real, 1, spolyv + norder * (norder - 1), 1)
                   + BlasConnector::dot(norder, p_che->coef_real, 1, spolyv + norder - 1, norder)
                   - last_coef * last_spolyv);
    }

#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    double relative_error = std::abs(error / ref);
    GlobalV::ofs_running << info << "Relative Chebyshev Precision: " << relative_error * 1e9 << "E-09" << std::endl;
    if (relative_error > thr)
    {
        std::stringstream ss;
        ss << relative_error;
        std::string fractxt, tartxt;
        ss >> fractxt;
        ss.clear();
        ss << thr;
        ss >> tartxt;
        std::string warningtxt = "( " + info + " relative Chebyshev error = " + fractxt + " > threshold = " + tartxt
                                 + " ) Maybe you should increase the parameter \"nche_sto\" for more accuracy.";
        ModuleBase::WARNING("Stochastic_Chebychev", warningtxt);
    }
    //===============================
}

void Stochastic_Iter::itermu(const int iter, elecstate::ElecState* pes)
{
    ModuleBase::TITLE("Stochastic_Iter", "itermu");
    ModuleBase::timer::tick("Stochastic_Iter", "itermu");
    double dmu;
    if (iter == 1)
    {
        dmu = 2;
        th_ne = 0.1 * PARAM.inp.scf_thr * PARAM.inp.nelec;
        // std::cout<<"th_ne "<<th_ne<<std::endl;
    }
    else
    {
        dmu = 0.1;
        th_ne = 1e-2 * PARAM.inp.scf_thr * PARAM.inp.nelec;
        th_ne = std::min(th_ne, 1e-5);
    }
    this->stofunc.mu = mu0 - dmu;
    double ne1 = calne(pes);
    double mu1 = this->stofunc.mu;

    this->stofunc.mu = mu0 + dmu;
    double ne2 = calne(pes);
    double mu2 = this->stofunc.mu;
    double Dne = th_ne + 1;
    double ne3;
    double mu3;

    while (ne1 > targetne)
    {
        mu2 = mu1;
        mu1 -= dmu;
        this->stofunc.mu = mu1;
        ne1 = calne(pes);
        // std::cout<<"Reset mu1 from "<<mu1+dmu<<" to "<<mu1<<std::endl;
        dmu *= 2;
    }
    while (ne2 < targetne)
    {
        mu1 = mu2;
        mu2 += dmu;
        this->stofunc.mu = mu2;
        ne2 = calne(pes);
        // cout<<"Reset mu2 from "<<mu2-dmu<<" to "<<mu2<<endl;
        dmu *= 2;
    }
    int count = 0;
    while (Dne > th_ne)
    {
        mu3 = (mu2 + mu1) / 2;
        this->stofunc.mu = mu3;
        ne3 = calne(pes);
        if (ne3 < targetne)
        {
            mu1 = mu3;
        }
        else if (ne3 > targetne)
        {
            mu2 = mu3;
        }
        Dne = std::abs(targetne - ne3);

        count++;
        if (count > 60)
        {
            std::cout << "Fermi energy cannot be converged. Set THNE to " << th_ne << std::endl;
            th_ne *= 1e1;
            if (th_ne > 1e1)
            {
                ModuleBase::WARNING_QUIT("Stochastic_Iter",
                                         "Cannot converge feimi energy. Please retry with different random number");
            }
        }
    }
    pes->eferm.ef = this->stofunc.mu = mu0 = mu3;
    GlobalV::ofs_running << "Converge fermi energy = " << mu3 << " Ry in " << count << " steps." << std::endl;
    this->check_precision(targetne, 10 * PARAM.inp.scf_thr, "Ne");

    // Set wf.wg
    if (PARAM.inp.nbands > 0)
    {
        for (int ikk = 0; ikk < this->pkv->get_nks(); ++ikk)
        {
            double* en = &pes->ekb(ikk, 0);
            for (int iksb = 0; iksb < PARAM.inp.nbands; ++iksb)
            {
                pes->wg(ikk, iksb) = stofunc.fd(en[iksb]) * this->pkv->wk[ikk];
            }
        }
    }
    ModuleBase::timer::tick("Stochastic_Iter", "itermu");
    return;
}

void Stochastic_Iter::calPn(const int& ik, Stochastic_WF& stowf)
{
    ModuleBase::TITLE("Stochastic_Iter", "calPn");
    ModuleBase::timer::tick("Stochastic_Iter", "calPn");

    const int norder = p_che->norder;
    const int nchip_ik = nchip[ik];
    const int npw = stowf.ngk[ik];
    const int npwx = stowf.npwx;
    if (ik == 0)
    {
        if (this->method == 1)
        {
            ModuleBase::GlobalFunc::ZEROS(spolyv, norder);
        }
        else
        {
            ModuleBase::GlobalFunc::ZEROS(spolyv, norder * norder);
        }
    }
    std::complex<double>* pchi;
    if (PARAM.inp.nbands > 0)
    {
        stowf.chiortho->fix_k(ik);
        pchi = stowf.chiortho->get_pointer();
    }
    else
    {
        stowf.chi0->fix_k(ik);
        pchi = stowf.chi0->get_pointer();
    }

    auto hchi_norm = std::bind(&Stochastic_hchi::hchi_norm,
                               &stohchi,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               std::placeholders::_3);
    if (this->method == 1)
    {
        p_che->tracepolyA(hchi_norm, pchi, npw, npwx, nchip_ik);
        for (int i = 0; i < norder; ++i)
        {
            spolyv[i] += p_che->polytrace[i] * this->pkv->wk[ik];
        }
    }
    else
    {
        p_che->calpolyvec_complex(hchi_norm, pchi, stowf.chiallorder[ik].c, npw, npwx, nchip_ik);
        double* vec_all = (double*)stowf.chiallorder[ik].c;
        char trans = 'T';
        char normal = 'N';
        double one = 1;
        int LDA = npwx * nchip_ik * 2;
        int M = npwx * nchip_ik * 2; // Do not use kv.ngk[ik]
        int N = norder;
        double kweight = this->pkv->wk[ik];
        dgemm_(&trans, &normal, &N, &N, &M, &kweight, vec_all, &LDA, vec_all, &LDA, &one, spolyv, &N);
    }
    ModuleBase::timer::tick("Stochastic_Iter", "calPn");
    return;
}

double Stochastic_Iter::calne(elecstate::ElecState* pes)
{
    ModuleBase::timer::tick("Stochastic_Iter", "calne");
    double totne = 0;
    KS_ne = 0;
    const int norder = p_che->norder;
    double sto_ne;
    if (this->method == 1)
    {
        // Note: spolyv contains kv.wk[ik]
        auto nfd = std::bind(&Sto_Func<double>::nfd, &this->stofunc, std::placeholders::_1);
        p_che->calcoef_real(nfd);
        sto_ne = BlasConnector::dot(norder, p_che->coef_real, 1, spolyv, 1);
    }
    else
    {
        auto nroot_fd = std::bind(&Sto_Func<double>::nroot_fd, &this->stofunc, std::placeholders::_1);
        p_che->calcoef_real(nroot_fd);
        sto_ne = vTMv(p_che->coef_real, spolyv, norder);
    }
    if (PARAM.inp.nbands > 0)
    {
        for (int ikk = 0; ikk < this->pkv->get_nks(); ++ikk)
        {
            double* en = &pes->ekb(ikk, 0);
            for (int iksb = 0; iksb < PARAM.inp.nbands; ++iksb)
            {
                KS_ne += stofunc.fd(en[iksb]) * this->pkv->wk[ikk];
            }
        }
    }
    KS_ne /= GlobalV::NPROC_IN_POOL;
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &KS_ne, 1, MPI_DOUBLE, MPI_SUM, STO_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sto_ne, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    totne = KS_ne + sto_ne;
    ModuleBase::timer::tick("Stochastic_Iter", "calne");
    return totne;
}

void Stochastic_Iter::calHsqrtchi(Stochastic_WF& stowf)
{
    auto nroot_fd = std::bind(&Sto_Func<double>::nroot_fd, &this->stofunc, std::placeholders::_1);
    p_che->calcoef_real(nroot_fd);
    for (int ik = 0; ik < this->pkv->get_nks(); ++ik)
    {
        // init k
        if (this->pkv->get_nks() > 1)
        {

            if (GlobalC::ppcell.nkb > 0
                && (PARAM.inp.basis_type == "pw"
                    || PARAM.inp.basis_type == "lcao_in_pw")) // xiaohui add 2013-09-02. Attention...
            {
                GlobalC::ppcell.getvnl(ik, GlobalC::ppcell.vkb);
            }

        }
        stohchi.current_ik = ik;

        this->calTnchi_ik(ik, stowf);
    }
}

void Stochastic_Iter::sum_stoband(Stochastic_WF& stowf,
                                  elecstate::ElecState* pes,
                                  hamilt::Hamilt<std::complex<double>>* pHamilt,
                                  ModulePW::PW_Basis_K* wfc_basis)
{
    ModuleBase::TITLE("Stochastic_Iter", "sum_stoband");
    ModuleBase::timer::tick("Stochastic_Iter", "sum_stoband");
    int nrxx = wfc_basis->nrxx;
    int npwx = wfc_basis->npwk_max;
    const int norder = p_che->norder;

    //---------------cal demet-----------------------
    double stodemet;
    if (this->method == 1)
    {
        auto nfdlnfd = std::bind(&Sto_Func<double>::nfdlnfd, &this->stofunc, std::placeholders::_1);
        p_che->calcoef_real(nfdlnfd);
        stodemet = BlasConnector::dot(norder, p_che->coef_real, 1, spolyv, 1);
    }
    else
    {
        auto nroot_fdlnfd = std::bind(&Sto_Func<double>::n_root_fdlnfd, &this->stofunc, std::placeholders::_1);
        p_che->calcoef_real(nroot_fdlnfd);
        stodemet = -vTMv(p_che->coef_real, spolyv, norder);
    }

    if (PARAM.inp.nbands > 0)
    {
        for (int ikk = 0; ikk < this->pkv->get_nks(); ++ikk)
        {
            double* enb = &pes->ekb(ikk, 0);
            // number of electrons in KS orbitals
            for (int iksb = 0; iksb < PARAM.inp.nbands; ++iksb)
            {
                pes->f_en.demet += stofunc.fdlnfd(enb[iksb]) * this->pkv->wk[ikk];
            }
        }
    }
    pes->f_en.demet /= GlobalV::NPROC_IN_POOL;
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &pes->f_en.demet, 1, MPI_DOUBLE, MPI_SUM, STO_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &stodemet, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    pes->f_en.demet += stodemet;
    this->check_precision(pes->f_en.demet, 1e-4, "TS");
    pes->f_en.demet *= Occupy::gaussian_parameter;

    //--------------------cal eband------------------------
    double sto_eband = 0;
    if (this->method == 1)
    {
        auto nxfd = std::bind(&Sto_Func<double>::nxfd, &this->stofunc, std::placeholders::_1);
        p_che->calcoef_real(nxfd);
        sto_eband = BlasConnector::dot(norder, p_che->coef_real, 1, spolyv, 1);
    }
    else
    {
        for (int ik = 0; ik < this->pkv->get_nks(); ++ik)
        {
            const int nchip_ik = nchip[ik];
            if (this->pkv->get_nks() > 1)
            {
                pHamilt->updateHk(ik);
                stowf.shchi->fix_k(ik);
            }
            stohchi.current_ik = ik;
            const int npw = this->pkv->ngk[ik];
            const double kweight = this->pkv->wk[ik];
            std::complex<double>* hshchi = new std::complex<double>[nchip_ik * npwx];
            std::complex<double>* tmpin = stowf.shchi->get_pointer();
            std::complex<double>* tmpout = hshchi;
            stohchi.hchi(tmpin, tmpout, nchip_ik);
            for (int ichi = 0; ichi < nchip_ik; ++ichi)
            {
                sto_eband += kweight * ModuleBase::GlobalFunc::ddot_real(npw, tmpin, tmpout, false);
                tmpin += npwx;
                tmpout += npwx;
            }
            delete[] hshchi;
        }
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &sto_eband, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    pes->f_en.eband += sto_eband;
    //---------------------cal rho-------------------------
    double* sto_rho = new double[nrxx];

    double dr3 = GlobalC::ucell.omega / wfc_basis->nxyz;
    double tmprho, tmpne;
    std::complex<double> outtem;
    double sto_ne = 0;
    ModuleBase::GlobalFunc::ZEROS(sto_rho, nrxx);

    std::complex<double>* porter = new std::complex<double>[nrxx];
    double out2;

    double* ksrho = nullptr;
    if (PARAM.inp.nbands > 0 && GlobalV::MY_STOGROUP == 0)
    {
        ksrho = new double[nrxx];
        ModuleBase::GlobalFunc::DCOPY(pes->charge->rho[0], ksrho, nrxx);
        ModuleBase::GlobalFunc::ZEROS(pes->charge->rho[0], nrxx);
    }

    for (int ik = 0; ik < this->pkv->get_nks(); ++ik)
    {
        const int nchip_ik = nchip[ik];
        stowf.shchi->fix_k(ik);
        std::complex<double>* tmpout = stowf.shchi->get_pointer();
        for (int ichi = 0; ichi < nchip_ik; ++ichi)
        {
            wfc_basis->recip2real(tmpout, porter, ik);
            for (int ir = 0; ir < nrxx; ++ir)
            {
                pes->charge->rho[0][ir] += norm(porter[ir]) * this->pkv->wk[ik];
            }
            tmpout += npwx;
        }
    }
    delete[] porter;
#ifdef __MPI
    // temporary, rho_mpi should be rewrite as a tool function! Now it only treats pes->charge->rho
    pes->charge->rho_mpi();
#endif
    for (int ir = 0; ir < nrxx; ++ir)
    {
        tmprho = pes->charge->rho[0][ir] / GlobalC::ucell.omega;
        sto_rho[ir] = tmprho;
        sto_ne += tmprho;
    }
    sto_ne *= dr3;

#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &sto_ne, 1, MPI_DOUBLE, MPI_SUM, POOL_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sto_ne, 1, MPI_DOUBLE, MPI_SUM, PARAPW_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sto_rho, nrxx, MPI_DOUBLE, MPI_SUM, PARAPW_WORLD);
#endif
    double factor = targetne / (KS_ne + sto_ne);
    if (std::abs(factor - 1) > 1e-10)
    {
        GlobalV::ofs_running << "Renormalize rho from ne = " << sto_ne + KS_ne << " to targetne = " << targetne
                             << std::endl;
    }
    else
    {
        factor = 1;
    }

    if (GlobalV::MY_STOGROUP == 0)
    {
        if (PARAM.inp.nbands > 0)
        {
            ModuleBase::GlobalFunc::DCOPY(ksrho, pes->charge->rho[0], nrxx);
        }
        else
        {
            ModuleBase::GlobalFunc::ZEROS(pes->charge->rho[0], nrxx);
        }
    }

    if (GlobalV::MY_STOGROUP == 0)
    {
        for (int is = 0; is < 1; ++is)
        {
            for (int ir = 0; ir < nrxx; ++ir)
            {
                pes->charge->rho[is][ir] += sto_rho[ir];
                pes->charge->rho[is][ir] *= factor;
            }
        }
    }
    delete[] sto_rho;
    delete[] ksrho;
    ModuleBase::timer::tick("Stochastic_Iter", "sum_stoband");
    return;
}

void Stochastic_Iter::calTnchi_ik(const int& ik, Stochastic_WF& stowf)
{
    const int npw = stowf.ngk[ik];
    const int npwx = stowf.npwx;
    stowf.shchi->fix_k(ik);
    std::complex<double>* out = stowf.shchi->get_pointer();
    std::complex<double>* pchi;
    if (PARAM.inp.nbands > 0)
    {
        stowf.chiortho->fix_k(ik);
        pchi = stowf.chiortho->get_pointer();
    }
    else
    {
        stowf.chi0->fix_k(ik);
        pchi = stowf.chi0->get_pointer();
    }
    if (this->method == 2)
    {
        char transa = 'N';
        std::complex<double> one = 1;
        int inc = 1;
        std::complex<double> zero = 0;
        int LDA = npwx * nchip[ik];
        int M = npwx * nchip[ik];
        int N = p_che->norder;
        std::complex<double>* coef_real = new std::complex<double>[p_che->norder];
        for (int i = 0; i < p_che->norder; ++i)
        {
            coef_real[i] = p_che->coef_real[i];
        }
        zgemv_(&transa, &M, &N, &one, stowf.chiallorder[ik].c, &LDA, coef_real, &inc, &zero, out, &inc);
        delete[] coef_real;
    }
    else
    {
        auto hchi_norm = std::bind(&Stochastic_hchi::hchi_norm,
                                   &stohchi,
                                   std::placeholders::_1,
                                   std::placeholders::_2,
                                   std::placeholders::_3);
        p_che->calfinalvec_real(hchi_norm, pchi, out, npw, npwx, nchip[ik]);
    }
}
