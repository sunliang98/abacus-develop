#include "ml_data.h"

#include "npy.hpp"
#include "module_base/parallel_reduce.h"
#include "module_base/global_function.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_base/module_device/types.h"

void ML_data::set_para(
    const int &nx,
    const double &nelec, 
    const double &tf_weight, 
    const double &vw_weight,
    const double &chi_p,
    const double &chi_q,
    const std::string &chi_xi_,
    const std::string &chi_pnl_,
    const std::string &chi_qnl_,
    const int &nkernel,
    const std::string &kernel_type_,
    const std::string &kernel_scaling_,
    const std::string &yukawa_alpha_,
    const std::string &kernel_file_,
    ModulePW::PW_Basis *pw_rho
)
{
    this->nx = nx;
    this->nkernel = nkernel;
    this->chi_p = chi_p;
    this->chi_q = chi_q;
    this->split_string(chi_xi_, nkernel, 1., this->chi_xi);
    this->split_string(chi_pnl_, nkernel, 1., this->chi_pnl);
    this->split_string(chi_qnl_, nkernel, 1., this->chi_qnl);

    this->split_string(kernel_type_, nkernel, 1, this->kernel_type);
    this->split_string(kernel_scaling_, nkernel, 1., this->kernel_scaling);
    this->split_string(yukawa_alpha_, nkernel, 1., this->yukawa_alpha);
    std::string temp = "none";
    this->split_string(kernel_file_, nkernel, temp, this->kernel_file);
    std::cout << "nkernel    " << nkernel << std::endl;

    if (GlobalV::of_wt_rho0 != 0)
    {
        this->rho0 = GlobalV::of_wt_rho0;
    }
    else
    {
        this->rho0 = 1./GlobalC::ucell.omega * nelec;
    }

    this->kF = pow(3. * pow(ModuleBase::PI, 2) * this->rho0, 1./3.);
    this->tkF = 2. * this->kF;

    this->kernel = new double*[this->nkernel];
    for (int ik = 0; ik < this->nkernel; ++ik)
    {
        // delete[] this->kernel[ik];
        this->kernel[ik] = new double[pw_rho->npw];
        if (this->kernel_type[ik] == 3 || this->kernel_type[ik] == 4) // 3 for TKK Al, and 4 for TKK Si
        {
            this->read_kernel(this->kernel_file[ik], this->kernel_scaling[ik], pw_rho, this->kernel[ik]);
        }
        else
        {
            double eta = 0.;
            for (int ip = 0; ip < pw_rho->npw; ++ip)
            {
                eta = sqrt(pw_rho->gg[ip]) * pw_rho->tpiba / this->tkF * this->kernel_scaling[ik];
                if (this->kernel_type[ik] == 1)
                {
                    this->kernel[ik][ip] = std::pow(1. / this->kernel_scaling[ik], 3) * this->MLkernel(eta, tf_weight, vw_weight);
                }
                else if (this->kernel_type[ik] == 2)
                {
                    this->kernel[ik][ip] = this->MLkernel_yukawa(eta, this->yukawa_alpha[ik]);
                }
            }
        }
    }
}

void ML_data::generateTrainData_WT(
    const double * const *prho, 
    KEDF_WT &wt, 
    KEDF_TF &tf, 
    ModulePW::PW_Basis *pw_rho,
    const double* veff    
)
{
    std::vector<std::vector<double>> nablaRho(3, std::vector<double>(this->nx, 0.));

    this->generate_descriptor(prho, pw_rho, nablaRho);

    std::vector<double> container(this->nx);
    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape of container and containernl
    
    // enhancement factor of Pauli potential
    if (GlobalV::of_kinetic == "wt")
    {
        this->getF_WT(wt, tf, prho, pw_rho, container);
        npy::SaveArrayAsNumpy("enhancement.npy", false, 1, cshape, container);

        // Pauli potential
        this->getPauli_WT(wt, tf, prho, pw_rho, container);
        npy::SaveArrayAsNumpy("pauli.npy", false, 1, cshape, container);
    }

    for (int ir = 0; ir < this->nx; ++ir)
    {
        container[ir] = veff[ir];
    }
    npy::SaveArrayAsNumpy("veff.npy", false, 1, cshape, container);
}

void ML_data::generateTrainData_KS(
    psi::Psi<std::complex<double>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    const double* veff
)
{
    std::vector<std::vector<double>> nablaRho(3, std::vector<double>(this->nx, 0.));

    this->generate_descriptor(pelec->charge->rho, pw_rho, nablaRho);

    std::vector<double> container(this->nx);
    std::vector<double> containernl(this->nx);

    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape of container and containernl
    // enhancement factor of Pauli energy, and Pauli potential
    this->getF_KS1(psi, pelec, pw_psi, pw_rho, nablaRho, container, containernl);

    Symmetry_rho srho;

    Charge* ptempRho = new Charge();
    ptempRho->nspin = GlobalV::NSPIN;
    ptempRho->nrxx = this->nx;
    ptempRho->rho_core = pelec->charge->rho_core;
    ptempRho->rho = new double*[1];
    ptempRho->rho[0] = new double[this->nx];
    ptempRho->rhog = new std::complex<double>*[1];
    ptempRho->rhog[0] = new std::complex<double>[pw_rho->npw];

    for (int ir = 0; ir < this->nx; ++ir) ptempRho->rho[0][ir] = container[ir];
    srho.begin(0, *ptempRho, pw_rho, GlobalC::Pgrid, GlobalC::ucell.symm);
    for (int ir = 0; ir < this->nx; ++ir) container[ir] = ptempRho->rho[0][ir];

    for (int ir = 0; ir < this->nx; ++ir) ptempRho->rho[0][ir] = containernl[ir];
    srho.begin(0, *ptempRho, pw_rho, GlobalC::Pgrid, GlobalC::ucell.symm);
    for (int ir = 0; ir < this->nx; ++ir) containernl[ir] = ptempRho->rho[0][ir];

    npy::SaveArrayAsNumpy("enhancement.npy", false, 1, cshape, container);
    npy::SaveArrayAsNumpy("pauli.npy", false, 1, cshape, containernl);

    // enhancement factor of Pauli energy, and Pauli potential
    this->getF_KS2(psi, pelec, pw_psi, pw_rho, container, containernl);

    for (int ir = 0; ir < this->nx; ++ir) ptempRho->rho[0][ir] = container[ir];
    srho.begin(0, *ptempRho, pw_rho, GlobalC::Pgrid, GlobalC::ucell.symm);
    for (int ir = 0; ir < this->nx; ++ir) container[ir] = ptempRho->rho[0][ir];

    for (int ir = 0; ir < this->nx; ++ir) ptempRho->rho[0][ir] = containernl[ir];
    srho.begin(0, *ptempRho, pw_rho, GlobalC::Pgrid, GlobalC::ucell.symm);
    for (int ir = 0; ir < this->nx; ++ir) containernl[ir] = ptempRho->rho[0][ir];

    npy::SaveArrayAsNumpy("enhancement2.npy", false, 1, cshape, container);
    npy::SaveArrayAsNumpy("pauli2.npy", false, 1, cshape, containernl);

    for (int ir = 0; ir < this->nx; ++ir)
    {
        container[ir] = veff[ir];
    }
    npy::SaveArrayAsNumpy("veff.npy", false, 1, cshape, container);

    delete ptempRho;
}

void ML_data::generate_descriptor(
    const double * const *prho, 
    ModulePW::PW_Basis *pw_rho,
    std::vector<std::vector<double>> &nablaRho
)
{
    // container which will contain gamma, p, q in turn
    std::vector<double> container(this->nx);
    std::vector<double> new_container(this->nx);
    // container contains gammanl, pnl, qnl in turn
    std::vector<double> containernl(this->nx);
    std::vector<double> new_containernl(this->nx);

    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape of container and containernl

    // rho
    std::vector<double> rho(this->nx);
    for (int ir = 0; ir < this->nx; ++ir) rho[ir] = prho[0][ir];
    npy::SaveArrayAsNumpy("rho.npy", false, 1, cshape, rho);

    // gamma
    this->getGamma(prho, container);
    npy::SaveArrayAsNumpy("gamma.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->nkernel; ++ik)
    {
        int ktype = this->kernel_type[ik];
        double kscaling = this->kernel_scaling[ik];

        // gamma_nl
        this->getGammanl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name("gammanl", ktype, kscaling), false, 1, cshape, containernl);

        // xi = gamma_nl/gamma
        this->getXi(container, containernl, new_container);
        npy::SaveArrayAsNumpy(this->file_name("xi", ktype, kscaling), false, 1, cshape, new_container);

        // tanhxi = tanh(xi)
        this->getTanhXi(ik, container, containernl, new_container);
        npy::SaveArrayAsNumpy(this->file_name("tanhxi", ktype, kscaling), false, 1, cshape, new_container);

        // (tanhxi)_nl
        this->getTanhXi_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("tanhxi_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }

    // nabla rho
    this->getNablaRho(prho, pw_rho, nablaRho);
    npy::SaveArrayAsNumpy("nablaRhox.npy", false, 1, cshape, nablaRho[0]);
    npy::SaveArrayAsNumpy("nablaRhoy.npy", false, 1, cshape, nablaRho[1]);
    npy::SaveArrayAsNumpy("nablaRhoz.npy", false, 1, cshape, nablaRho[2]);

    // p
    this->getP(prho, pw_rho, nablaRho, container);
    npy::SaveArrayAsNumpy("p.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->nkernel; ++ik)
    {
        int ktype = this->kernel_type[ik];
        double kscaling = this->kernel_scaling[ik];

        // p_nl
        this->getPnl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name("pnl", ktype, kscaling), false, 1, cshape, containernl);

        // tanh(p_nl)
        this->getTanh_Pnl(ik, containernl, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("tanh_pnl", ktype, kscaling), false, 1, cshape, new_containernl);

        // tanh(p)
        this->getTanhP(container, new_container);
        npy::SaveArrayAsNumpy("tanhp.npy", false, 1, cshape, new_container);

        // tanh(p)_nl
        this->getTanhP_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("tanhp_nl", ktype, kscaling), false, 1, cshape, new_containernl);

        // f(p) = p/(1+p)
        this->getfP(container, new_container);
        npy::SaveArrayAsNumpy("fp.npy", false, 1, cshape, new_container);

        // f(p)_nl
        this->getfP_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("fp_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }

    // q
    this->getQ(prho, pw_rho, container);
    npy::SaveArrayAsNumpy("q.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->nkernel; ++ik)
    {
        int ktype = this->kernel_type[ik];
        double kscaling = this->kernel_scaling[ik];

        // q_nl
        this->getQnl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name("qnl", ktype, kscaling), false, 1, cshape, containernl);

        // tanh(q_nl)
        this->getTanh_Qnl(ik, containernl, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("tanh_qnl", ktype, kscaling), false, 1, cshape, new_containernl);

        // tanh(q)
        this->getTanhQ(container, new_container);
        npy::SaveArrayAsNumpy("tanhq.npy", false, 1, cshape, new_container);

        // tanh(q)_nl
        this->getTanhQ_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name("tanhq_nl", ktype, kscaling), false, 1, cshape, new_containernl);

        // // f(q) = q/(1+q)
        // this->getfQ(container, new_container);
        // npy::SaveArrayAsNumpy("fq.npy", false, 1, cshape, new_container);

        // // f(q)_nl
        // this->getfQ_nl(ik, new_container, pw_rho, new_containernl);
        // npy::SaveArrayAsNumpy(this->file_name("fq_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }

    this->get_r_matrix(GlobalC::ucell, pw_rho, container);
    npy::SaveArrayAsNumpy("r_min.npy", false, 1, cshape, container);
}

void ML_data::getGamma(const double * const *prho, std::vector<double> &rgamma)
{
    for(int ir = 0; ir < this->nx; ++ir)
    {
        rgamma[ir] = pow(prho[0][ir]/this->rho0, 1./3.);
    }
}

void ML_data::getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp)
{
    for(int ir = 0; ir < this->nx; ++ir)
    {
        rp[ir] = 0.;
        for (int j = 0; j < 3; ++j)
        {
            rp[ir] += pow(pnablaRho[j][ir], 2);
        }
        rp[ir] *= this->pqcoef / pow(prho[0][ir], 8.0/3.0);
    }
}

void ML_data::getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq)
{
    // get Laplacian rho
    std::complex<double> *recipRho = new std::complex<double>[pw_rho->npw];
    pw_rho->real2recip(prho[0], recipRho);
    for (int ip = 0; ip < pw_rho->npw; ++ip)
    {
        recipRho[ip] *= - pw_rho->gg[ip] * pw_rho->tpiba2;
    }
    pw_rho->recip2real(recipRho, rq.data());

    for (int ir = 0; ir < this->nx; ++ir)
    {
        rq[ir] *= this->pqcoef / pow(prho[0][ir], 5.0/3.0);
    }

    delete[] recipRho;
}

void ML_data::getGammanl(const int ikernel, std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl)
{
    this->multiKernel(ikernel, pgamma.data(), pw_rho, rgammanl.data());
}

void ML_data::getPnl(const int ikernel, std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl)
{
    this->multiKernel(ikernel, pp.data(), pw_rho, rpnl.data());
}

void ML_data::getQnl(const int ikernel, std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl)
{
    this->multiKernel(ikernel, pq.data(), pw_rho, rqnl.data());
}

// xi = gammanl/gamma
void ML_data::getXi(std::vector<double> &pgamma, std::vector<double> &pgammanl, std::vector<double> &rxi)
{
    for (int ir = 0; ir < this->nx; ++ir)
    {
        if (pgamma[ir] == 0)
        {
            std::cout << "WARNING: gamma=0" << std::endl;
            rxi[ir] = 0.;
        }
        else
        {
            rxi[ir] = pgammanl[ir]/pgamma[ir];
        }
    }
}

// tanhxi = tanh(gammanl/gamma)
void ML_data::getTanhXi(const int ikernel, std::vector<double> &pgamma, std::vector<double> &pgammanl, std::vector<double> &rtanhxi)
{
    for (int ir = 0; ir < this->nx; ++ir)
    {
        if (pgamma[ir] == 0)
        {
            std::cout << "WARNING: gamma=0" << std::endl;
            rtanhxi[ir] = 0.;
        }
        else
        {
            rtanhxi[ir] = std::tanh(pgammanl[ir]/pgamma[ir] * this->chi_xi[ikernel]);
        }
    }
}

// tanh(p)
void ML_data::getTanhP(std::vector<double> &pp, std::vector<double> &rtanhp)
{
    this->tanh(pp, rtanhp, this->chi_p);
}

// tanh(q)
void ML_data::getTanhQ(std::vector<double> &pq, std::vector<double> &rtanhq)
{
    this->tanh(pq, rtanhq, this->chi_q);
}

// tanh(pnl)
void ML_data::getTanh_Pnl(const int ikernel, std::vector<double> &ppnl, std::vector<double> &rtanh_pnl)
{
    this->tanh(ppnl, rtanh_pnl, this->chi_pnl[ikernel]);
}

// tanh(qnl)
void ML_data::getTanh_Qnl(const int ikernel, std::vector<double> &pqnl, std::vector<double> &rtanh_qnl)
{
    this->tanh(pqnl, rtanh_qnl, this->chi_qnl[ikernel]);
}

// tanh(p)_nl
void ML_data::getTanhP_nl(const int ikernel, std::vector<double> &ptanhp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhp_nl)
{
    this->multiKernel(ikernel, ptanhp.data(), pw_rho, rtanhp_nl.data());
}

// tanh(q)_nl
void ML_data::getTanhQ_nl(const int ikernel, std::vector<double> &ptanhq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhq_nl)
{
    this->multiKernel(ikernel, ptanhq.data(), pw_rho, rtanhq_nl.data());
}

// f(p) = p/(1+p)
void ML_data::getfP(std::vector<double> &pp, std::vector<double> &rfp)
{
    this->f(pp, rfp);
}

// f(q) = q/(1+q)
void ML_data::getfQ(std::vector<double> &pq, std::vector<double> &rfq)
{
    this->f(pq, rfq);
}

// f(p)_nl
void ML_data::getfP_nl(const int ikernel, std::vector<double> &pfp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rfp_nl)
{
    this->multiKernel(ikernel, pfp.data(), pw_rho, rfp_nl.data());
}

// f(q)_nl
void ML_data::getfQ_nl(const int ikernel, std::vector<double> &pfq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rfq_nl)
{
    this->multiKernel(ikernel, pfq.data(), pw_rho, rfq_nl.data());
}

// (tanhxi)_nl
void ML_data::getTanhXi_nl(const int ikernel, std::vector<double> &ptanhxi, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhxi_nl)
{
    this->multiKernel(ikernel, ptanhxi.data(), pw_rho, rtanhxi_nl.data());
}

void ML_data::getPauli_WT(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpauli)
{
    ModuleBase::matrix potential(1, this->nx, true);

    tf.tf_potential(prho, potential);
    wt.wt_potential(prho, pw_rho, potential);

    for (int ir = 0; ir < this->nx; ++ir) rpauli[ir] = potential(0, ir);
}

void ML_data::getF_WT(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rF)
{
    double wtden = 0.;
    double tfden = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        wtden = wt.get_energy_density(prho, 0, ir, pw_rho);
        tfden = tf.get_energy_density(prho, 0, ir);
        rF[ir] = 1. + wtden/tfden;
        // if (wtden < 0) std::cout << wtden/tfden << std::endl;
    }
}

void ML_data::getF_KS1(
    psi::Psi<std::complex<double>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    const std::vector<std::vector<double>> &nablaRho,
    std::vector<double> &rF,
    std::vector<double> &rpauli
)
{
    double *pauliED = new double[this->nx]; // Pauli Energy Density
    ModuleBase::GlobalFunc::ZEROS(pauliED, this->nx);

    double *pauliPot = new double[this->nx];
    ModuleBase::GlobalFunc::ZEROS(pauliPot, this->nx);

    std::complex<double> *wfcr = new std::complex<double>[this->nx];
    ModuleBase::GlobalFunc::ZEROS(wfcr, this->nx);

    double epsilonM = pelec->ekb(0,0);
    assert(GlobalV::NSPIN == 1);

    base_device::DEVICE_CPU* ctx;

    // calculate positive definite kinetic energy density
    for (int ik = 0; ik < psi->get_nk(); ++ik)
    {
        psi->fix_k(ik);
        int ikk = psi->get_current_k();
        assert(ikk == ik);
        int npw = psi->get_current_nbas();
        int nbands = psi->get_nbands();
        for (int ibnd = 0; ibnd < nbands; ++ibnd)
        {
            if (pelec->wg(ik, ibnd) < ModuleBase::threshold_wg) {
                continue;
            }

            pw_psi->recip_to_real(ctx, &psi->operator()(ibnd,0), wfcr, ik);
            const double w1 = pelec->wg(ik, ibnd) / GlobalC::ucell.omega;
            
            // output one wf, to check KS equation
            if (ik == 0 && ibnd == 0)
            {
                std::vector<double> wf_real = std::vector<double>(this->nx);
                std::vector<double> wf_imag = std::vector<double>(this->nx);
                for (int ir = 0; ir < this->nx; ++ir)
                {
                    wf_real[ir] = wfcr[ir].real();
                    wf_imag[ir] = wfcr[ir].imag();
                }
                const long unsigned cshape[] = {(long unsigned) this->nx}; // shape of container and containernl
                npy::SaveArrayAsNumpy("wfc_real.npy", false, 1, cshape, wf_real);
                npy::SaveArrayAsNumpy("wfc_imag.npy", false, 1, cshape, wf_imag);
                std::cout << "eigenvalue of wfc is " << pelec->ekb(ik, ibnd) << std::endl;
                std::cout << "wg of wfc is " << pelec->wg(ik, ibnd) << std::endl;
            }

            if (w1 != 0.0)
            {
                // Find the energy of HOMO
                if (pelec->ekb(ik,ibnd) > epsilonM)
                {
                    epsilonM = pelec->ekb(ik,ibnd);
                }
                // The last term of Pauli potential
                for (int ir = 0; ir < pelec->charge->nrxx; ir++)
                {
                    pauliPot[ir] -= w1 * pelec->ekb(ik,ibnd) * norm(wfcr[ir]);
                }
            }

            for (int j = 0; j < 3; ++j)
            {
                ModuleBase::GlobalFunc::ZEROS(wfcr, pelec->charge->nrxx);
                for (int ig = 0; ig < npw; ig++)
                {
                    double fact
                        = pw_psi->getgpluskcar(ik, ig)[j] * GlobalC::ucell.tpiba;
                    wfcr[ig] = psi->operator()(ibnd, ig) * complex<double>(0.0, fact);
                }

                // pw_psi->recip_to_real(dev, wfcr, wfcr, ik);

                pw_psi->recip2real(wfcr, wfcr, ik);
                
                for (int ir = 0; ir < this->nx; ++ir)
                {
                    pauliED[ir] += w1 * norm(wfcr[ir]); // actually, here should be w1/2 * norm(wfcr[ir]), but we multiply 2 to convert Ha to Ry.
                }
            }
        }
    }

    std::cout << "(1) epsilon max = " << epsilonM << std::endl;
    // calculate the positive definite vW energy density
    for (int j = 0; j < 3; ++j)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            pauliED[ir] -= nablaRho[j][ir] * nablaRho[j][ir] / (8. * pelec->charge->rho[0][ir]) * 2.; // convert Ha to Ry.
        }
    }

    // // check the vW energy density
    // double **phi = new double*[1];
    // phi[0] = new double[this->nx];
    // std::vector<std::vector<double> > nablaPhi(3, std::vector<double>(this->nx, 0.));

    // for (int ir = 0; ir < this->nx; ++ir)
    // {
    //     phi[0][ir] = sqrt(pelec->charge->rho[0][ir]);
    // }
    // this->getNablaRho(phi, pw_rho, nablaPhi);

    // for (int ir = 0; ir < this->nx; ++ir)
    // {
    //     double term1 = nablaPhi[0][ir] * nablaPhi[0][ir] + nablaPhi[1][ir] * nablaPhi[1][ir] + nablaPhi[2][ir] * nablaPhi[2][ir];
    //     double term2 = (nablaRho[0][ir] * nablaRho[0][ir] + nablaRho[1][ir] * nablaRho[1][ir] + nablaRho[2][ir] * nablaRho[2][ir] )/ (8. * pelec->charge->rho[0][ir]) * 2.;
    //     if (term1 >= 1e-5 && term2 >= 1e-5)
    //     {
    //         std::cout << "1:" << term1 << "  2:" << term2 << std::endl;
    //         assert(fabs(term1 - term2) <= 1e-6);
    //     }
    // }
    // // ===========================

    for (int ir = 0; ir < this->nx; ++ir)
    {
        rF[ir] = pauliED[ir] / (this->cTF * pow(pelec->charge->rho[0][ir], 5./3.));
        // rpauli[ir] = pauliPot[ir];
        rpauli[ir] = (pauliED[ir] + pauliPot[ir])/pelec->charge->rho[0][ir] + epsilonM;
    }
}

void ML_data::getF_KS2(
    psi::Psi<std::complex<double>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    std::vector<double> &rF,
    std::vector<double> &rpauli
)
{
    double *pauliED = new double[this->nx]; // Pauli Energy Density
    ModuleBase::GlobalFunc::ZEROS(pauliED, this->nx);

    double *pauliPot = new double[this->nx];
    ModuleBase::GlobalFunc::ZEROS(pauliPot, this->nx);

    std::complex<double> *wfcr = new std::complex<double>[this->nx];
    ModuleBase::GlobalFunc::ZEROS(wfcr, this->nx);

    std::complex<double> *wfcg = nullptr;
    std::complex<double> *LapWfcr = new std::complex<double>[this->nx];
    ModuleBase::GlobalFunc::ZEROS(LapWfcr, this->nx);

    double epsilonM = pelec->ekb(0,0);
    assert(GlobalV::NSPIN == 1);

    base_device::DEVICE_CPU* ctx;

    // calculate kinetic energy density
    for (int ik = 0; ik < psi->get_nk(); ++ik)
    {
        psi->fix_k(ik);
        int ikk = psi->get_current_k();
        assert(ikk == ik);
        int npw = psi->get_current_nbas();
        int nbands = psi->get_nbands();
        delete[] wfcg;
        wfcg = new std::complex<double>[npw];
        for (int ibnd = 0; ibnd < nbands; ++ibnd)
        {
            if (pelec->wg(ik, ibnd) < ModuleBase::threshold_wg) {
                continue;
            }

            pw_psi->recip_to_real(ctx, &psi->operator()(ibnd,0), wfcr, ik);
            const double w1 = pelec->wg(ik, ibnd) / GlobalC::ucell.omega;

            // if (w1 != 0.0)
            // {
                // Find the energy of HOMO
                if (pelec->ekb(ik,ibnd) > epsilonM)
                {
                    epsilonM = pelec->ekb(ik,ibnd);
                }
                // The last term of Pauli potential
                for (int ir = 0; ir < pelec->charge->nrxx; ir++)
                {
                    // pauliPot[ir] += w1 * norm(wfcr[ir]);
                    pauliPot[ir] -= w1 * pelec->ekb(ik,ibnd) * norm(wfcr[ir]);
                }
            // }

            ModuleBase::GlobalFunc::ZEROS(wfcg, npw);
            for (int ig = 0; ig < npw; ig++)
            {
                double fact = pw_psi->getgk2(ik, ig) * GlobalC::ucell.tpiba2;
                wfcg[ig] = - psi->operator()(ibnd, ig) * fact;
            }

            pw_psi->recip2real(wfcg, LapWfcr, ik);
            
            for (int ir = 0; ir < this->nx; ++ir)
            {
                pauliED[ir] += - w1 * (conj(wfcr[ir]) * LapWfcr[ir]).real(); // actually, here should be w1/2 * norm(wfcr[ir]), but we multiply 2 to convert Ha to Ry.
            }
        }
    }

    std::cout << "(2) epsilon max = " << epsilonM << std::endl;
    // calculate the positive definite vW energy density
    double *phi = new double[this->nx];
    double *LapPhi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        phi[ir] = sqrt(pelec->charge->rho[0][ir]);
    }
    this->Laplacian(phi, pw_rho, LapPhi);

    for (int ir = 0; ir < this->nx; ++ir)
    {
        pauliED[ir] -= - phi[ir] * LapPhi[ir]; // convert Ha to Ry.
    }

    for (int ir = 0; ir < this->nx; ++ir)
    {
        rF[ir] = pauliED[ir] / (this->cTF * pow(pelec->charge->rho[0][ir], 5./3.));
        // rpauli[ir] = pauliPot[ir];
        rpauli[ir] = (pauliED[ir] + pauliPot[ir])/pelec->charge->rho[0][ir] + epsilonM;
    }
}

void ML_data::getNablaRho(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &rnablaRho)
{
    std::complex<double> *recipRho = new std::complex<double>[pw_rho->npw];
    std::complex<double> *recipNablaRho = new std::complex<double>[pw_rho->npw];
    pw_rho->real2recip(prho[0], recipRho);
    
    std::complex<double> img = 1.0j;
    for (int j = 0; j < 3; ++j)
    {
        for (int ip = 0; ip < pw_rho->npw; ++ip)
        {
            recipNablaRho[ip] = img * pw_rho->gcar[ip][j] * recipRho[ip] * pw_rho->tpiba;
        }
        pw_rho->recip2real(recipNablaRho, rnablaRho[j].data());
    }

    delete[] recipRho;
    delete[] recipNablaRho;
}

double ML_data::MLkernel(double eta, double tf_weight, double vw_weight)
{
    if (eta < 0.) 
    {
        return 0.;
    }
    // limit for small eta
    else if (eta < 1e-10)
    {
        return 1. - tf_weight + eta * eta * (1./3. - 3. * vw_weight);
    }
    // around the singularity
    else if (std::abs(eta - 1.) < 1e-10)
    {
        return 2. - tf_weight - 3. * vw_weight + 20. * (eta - 1);
    }
    // Taylor expansion for high eta
    else if (eta > 3.65)
    {
        double eta2 = eta * eta;
        double invEta2 = 1. / eta2;
        double LindG = 3. * (1. - vw_weight) * eta2 
                        -tf_weight-0.6 
                        + invEta2 * (-0.13714285714285712 
                        + invEta2 * (-6.39999999999999875E-2
                        + invEta2 * (-3.77825602968460128E-2
                        + invEta2 * (-2.51824061652633074E-2
                        + invEta2 * (-1.80879839616166146E-2
                        + invEta2 * (-1.36715733124818332E-2
                        + invEta2 * (-1.07236045520990083E-2
                        + invEta2 * (-8.65192783339199453E-3 
                        + invEta2 * (-7.1372762502456763E-3
                        + invEta2 * (-5.9945117538835746E-3
                        + invEta2 * (-5.10997527675418131E-3 
                        + invEta2 * (-4.41060829979912465E-3 
                        + invEta2 * (-3.84763737842981233E-3 
                        + invEta2 * (-3.38745061493813488E-3 
                        + invEta2 * (-3.00624946457977689E-3)))))))))))))));
        return LindG;
    }
    else
    {
        return 1. / (0.5 + 0.25 * (1. - eta * eta) / eta * std::log((1 + eta)/std::abs(1 - eta)))
                 - 3. * vw_weight * eta * eta - tf_weight;
    }
}

double ML_data::MLkernel_yukawa(double eta, double alpha)
{
    return (eta == 0 && alpha == 0) ? 0. : M_PI / (eta * eta + alpha * alpha / 4.);
}

// Read kernel from file
void ML_data::read_kernel(const std::string &fileName, const double& scaling, ModulePW::PW_Basis *pw_rho, double* kernel_)
{
    std::ifstream ifs(fileName.c_str(), std::ios::in);

    GlobalV::ofs_running << "Read WT kernel from " << fileName << std::endl;
    if (!ifs) ModuleBase::WARNING_QUIT("ml_data.cpp", "The kernel file not found");

    int kineType = 0;
    double kF_in = 0.;
    double rho0_in = 0.;
    int nq_in = 0;
    double maxEta_in = 0.;

    ifs >> kineType;
    ifs >> kF_in;
    ifs >> rho0_in;
    ifs >> nq_in;

    double *eta_in = new double[nq_in];
    double *w0_in = new double[nq_in];

    for (int iq = 0; iq < nq_in; ++iq)
    {
        ifs >> eta_in[iq] >> w0_in[iq];
    }

    maxEta_in = eta_in[nq_in-1];

    double eta = 0.;
    double maxEta = 0.;
    int ind1 = 0;
    int ind2 = 0;
    int ind_mid = 0;
    double fac1 = 0.;
    double fac2 = 0.;
    for (int ig = 0; ig < pw_rho->npw; ++ig)
    {
        eta = sqrt(pw_rho->gg[ig]) * pw_rho->tpiba / this->tkF;
        eta = eta * scaling;
        maxEta = std::max(eta, maxEta);

        if (eta <= eta_in[0])
            kernel_[ig] = w0_in[0];
        else if (eta > maxEta_in)
            kernel_[ig] = w0_in[nq_in-1];
        else
        {
            ind1 = 1;
            ind2 = nq_in;
            while (ind1 < ind2 - 1)
            {
                ind_mid = (ind1 + ind2)/2;
                if (eta > eta_in[ind_mid])
                {
                    ind1 = ind_mid;
                }
                else
                {
                    ind2 = ind_mid;
                }
            }
            fac1 = (eta_in[ind2] - eta)/(eta_in[ind2] - eta_in[ind1]);
            fac2 = (eta - eta_in[ind1])/(eta_in[ind2] - eta_in[ind1]);
            kernel_[ig] = fac1 * w0_in[ind1] + fac2 * w0_in[ind2];
            kernel_[ig] *= std::pow(1. / scaling, 3);
        }
    }

    if (maxEta > maxEta_in) ModuleBase::WARNING("kedf_wt.cpp", "Please increase the maximal eta value in KEDF kernel file");

    delete[] eta_in;
    delete[] w0_in;
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "FILL WT KERNEL");
}

void ML_data::multiKernel(const int ikernel, double *pinput, ModulePW::PW_Basis *pw_rho, double *routput)
{
    std::complex<double> *recipOutput = new std::complex<double>[pw_rho->npw];

    pw_rho->real2recip(pinput, recipOutput);
    for (int ip = 0; ip < pw_rho->npw; ++ip)
    {
        recipOutput[ip] *= this->kernel[ikernel][ip];
    }
    pw_rho->recip2real(recipOutput, routput);

    delete[] recipOutput;
}

void ML_data::Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput)
{
    std::complex<double> *recipContainer = new std::complex<double>[pw_rho->npw];

    pw_rho->real2recip(pinput, recipContainer);
    for (int ip = 0; ip < pw_rho->npw; ++ip)
    {
        recipContainer[ip] *= - pw_rho->gg[ip] * pw_rho->tpiba2;
    }
    pw_rho->recip2real(recipContainer, routput);

    delete[] recipContainer;
}

void ML_data::divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput)
{
    std::complex<double> *recipContainer = new std::complex<double>[pw_rho->npw];
    std::complex<double> img = 1.0j;
    ModuleBase::GlobalFunc::ZEROS(routput, this->nx);
    for (int i = 0; i < 3; ++i)
    {
        pw_rho->real2recip(pinput[i], recipContainer);
        for (int ip = 0; ip < pw_rho->npw; ++ip)
        {
            recipContainer[ip] = img * pw_rho->gcar[ip][i] * pw_rho->tpiba * recipContainer[ip];
        }
        pw_rho->recip2real(recipContainer, routput, true);
    }

    delete[] recipContainer;
}

// void ML_data::dumpTensor(const torch::Tensor &data, std::string filename)
// {
//     std::cout << "Dumping " << filename << std::endl;
//     std::vector<double> v(this->nx);
//     for (int ir = 0; ir < this->nx; ++ir) v[ir] = data[ir].item<double>();
//     // std::vector<double> v(data.data_ptr<float>(), data.data_ptr<float>() + data.numel()); // this works, but only supports float tensor
//     const long unsigned cshape[] = {(long unsigned) this->nx}; // shape
//     npy::SaveArrayAsNumpy(filename, false, 1, cshape, v);
// }
void ML_data::loadVector(std::string filename, std::vector<double> &data)
{
    std::vector<long unsigned int> cshape = {(long unsigned) this->nx};
    bool fortran_order = false;
    npy::LoadArrayFromNumpy(filename, cshape, fortran_order, data);
}
void ML_data::dumpVector(std::string filename, const std::vector<double> &data)
{
    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape
    npy::SaveArrayAsNumpy(filename, false, 1, cshape, data);
}

void ML_data::tanh(std::vector<double> &pinput, std::vector<double> &routput, double chi)
{
    for (int i = 0; i < this->nx; ++i)
    {
        routput[i] = std::tanh(pinput[i] * chi);
    }
}

double ML_data::dtanh(double tanhx, double chi)
{
    return (1. - tanhx * tanhx) * chi;
}

void ML_data::f(std::vector<double> &pinput, std::vector<double> &routput)
{
    for (int i = 0; i < this->nx; ++i)
    {
        assert(pinput[i] >= 0);
        routput[i] = pinput[i]/(1. + pinput[i]);
    }
}

std::string ML_data::file_name(std::string parameter, const int kernel_type, const double kernel_scaling)
{
    std::stringstream ss;
    ss << "./" << parameter << "_" << kernel_type << "_" << kernel_scaling << ".npy";
    return ss.str();
}