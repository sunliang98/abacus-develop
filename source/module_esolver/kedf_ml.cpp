#include "./kedf_ml.h"
#include "npy.hpp"
#include "../src_parallel/parallel_reduce.h"


void KEDF_ML::set_para(int nx, double dV, double nelec, double tf_weight, double vw_weight, ModulePW::PW_Basis *pw_rho)
{
    this->nx = nx;
    this->dV = dV;
    this->nn = new NN_OF(this->nx, 6);

    if (GlobalV::of_wt_rho0 != 0)
    {
        this->rho0 = GlobalV::of_wt_rho0;
    }
    else
    {
        this->rho0 = 1./(pw_rho->nxyz * dV) * nelec;
    }

    this->kF = pow(3. * pow(ModuleBase::PI, 2) * this->rho0, 1./3.);
    this->tkF = 2. * this->kF;

    if (this->kernel != NULL) delete[] this->kernel;
    this->kernel = new double[pw_rho->npw];
    double eta = 0.;
    for (int ip = 0; ip < pw_rho->npw; ++ip)
    {
        eta = sqrt(pw_rho->gg[ip]) * pw_rho->tpiba / this->tkF;
        this->kernel[ip] = this->MLkernel(eta, tf_weight, vw_weight);
    }
}

double KEDF_ML::get_energy(const double * const * prho, KEDF_WT &wt, KEDF_TF &tf,  ModulePW::PW_Basis *pw_rho)
{
    std::vector<double> gamma(this->nx);
    std::vector<double> gammanl(this->nx);
    std::vector<double> p(this->nx);
    std::vector<double> pnl(this->nx);
    std::vector<double> q(this->nx);
    std::vector<double> qnl(this->nx);
    std::vector<std::vector<double>> nablaRho(3);
    for (int j = 0; j < 3; ++j) nablaRho[j] = std::vector<double>(this->nx);

    this->getGamma(prho, gamma);
    this->getGammanl(gamma, pw_rho, gammanl);
    this->getNablaRho(prho, pw_rho, nablaRho);
    this->getP(prho, pw_rho, nablaRho, p);
    this->getPnl(p, pw_rho, pnl);
    this->getQ(prho, pw_rho, q);
    this->getQnl(q, pw_rho, qnl);

    double energy = 0.;
    int ninpt = 6; ///////
    torch::Tensor inpt = torch::zeros(ninpt);
    for (int ir = 0; ir < this->nx; ++ir)
    {
        inpt[0] = gamma[ir];
        inpt[1] = gammanl[ir];
        inpt[2] = p[ir];
        inpt[3] = pnl[ir];
        inpt[4] = q[ir];
        inpt[5] = qnl[ir];

        energy += this->nn->forward(inpt).item<double>() * pow(prho[0][ir], 5./3.);
    }
    energy *= this->dV * this->cTF;
    this->MLenergy = energy;
    Parallel_Reduce::reduce_double_all(this->MLenergy);
    return energy;
}

void KEDF_ML::generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf,  ModulePW::PW_Basis *pw_rho)
{
    // container which will contain gamma, p, q in turn
    std::vector<double> container(this->nx);
    // container contains gammanl, pnl, qnl in turn
    std::vector<double> containernl(this->nx);
    // nabla rho
    std::vector<std::vector<double>> nablaRho(3);
    for (int j = 0; j < 3; ++j) nablaRho[j] = std::vector<double>(this->nx);

    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape of container and containernl

    // rho
    std::vector<double> rho(this->nx);
    for (int ir = 0; ir < this->nx; ++ir) rho[ir] = prho[0][ir];
    npy::SaveArrayAsNumpy("rho.npy", false, 1, cshape, rho);

    // gamma
    this->getGamma(prho, container);
    npy::SaveArrayAsNumpy("gamma.npy", false, 1, cshape, container);

    // gamma_nl
    this->getGammanl(container, pw_rho, containernl);
    npy::SaveArrayAsNumpy("gammanl.npy", false, 1, cshape, containernl);

    // nabla rho
    this->getNablaRho(prho, pw_rho, nablaRho);
    npy::SaveArrayAsNumpy("nablaRhox.npy", false, 1, cshape, nablaRho[0]);
    npy::SaveArrayAsNumpy("nablaRhoy.npy", false, 1, cshape, nablaRho[1]);
    npy::SaveArrayAsNumpy("nablaRhoz.npy", false, 1, cshape, nablaRho[2]);

    // p
    this->getP(prho, pw_rho, nablaRho, container);
    npy::SaveArrayAsNumpy("p.npy", false, 1, cshape, container);

    // p_nl
    this->getPnl(container, pw_rho, containernl);
    npy::SaveArrayAsNumpy("pnl.npy", false, 1, cshape, containernl);

    // q
    this->getQ(prho, pw_rho, container);
    npy::SaveArrayAsNumpy("q.npy", false, 1, cshape, container);

    // q_nl
    this->getQnl(container, pw_rho, containernl);
    npy::SaveArrayAsNumpy("qnl.npy", false, 1, cshape, containernl);

    // Pauli potential
    this->getPauli(wt, tf, prho, pw_rho, container);
    npy::SaveArrayAsNumpy("pauli.npy", false, 1, cshape, container);

    // enhancement factor of Pauli potential
    this->getF(wt, tf, prho, pw_rho, container);
    npy::SaveArrayAsNumpy("enhancement.npy", false, 1, cshape, container);
}

void KEDF_ML::getGamma(const double * const *prho, std::vector<double> &rgamma)
{
    for(int ir = 0; ir < this->nx; ++ir)
    {
        rgamma[ir] = pow(prho[0][ir]/this->rho0, 1./3.);
    }
}

void KEDF_ML::getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp)
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

void KEDF_ML::getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq)
{
    // get Laplacian rho
    std::complex<double> *recipRho = new std::complex<double>[pw_rho->npw];
    pw_rho->real2recip(prho[0], recipRho);
    for (int ik = 0; ik < pw_rho->npw; ++ik)
    {
        recipRho[ik] *= - pw_rho->gg[ik] * pw_rho->tpiba2;
    }
    pw_rho->recip2real(recipRho, rq.data());

    for (int ir = 0; ir < this->nx; ++ir)
    {
        rq[ir] *= this->pqcoef / pow(prho[0][ir], 5.0/3.0);
    }

    delete[] recipRho;
}

void KEDF_ML::getGammanl(std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl)
{
    this->multiKernel(pgamma.data(), pw_rho, rgammanl.data());
}

void KEDF_ML::getPnl(std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl)
{
    this->multiKernel(pp.data(), pw_rho, rpnl.data());
}

void KEDF_ML::getQnl(std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl)
{
    this->multiKernel(pq.data(), pw_rho, rqnl.data());
}

void KEDF_ML::getPauli(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpauli)
{
    ModuleBase::matrix potential(1, this->nx, true);

    tf.tf_potential(prho, potential);
    wt.WT_potential(prho, pw_rho, potential);

    for (int ir = 0; ir < this->nx; ++ir) rpauli[ir] = potential(0, ir);
}

void KEDF_ML::getF(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rF)
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

void KEDF_ML::getNablaRho(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &rnablaRho)
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

double KEDF_ML::MLkernel(double eta, double tf_weight, double vw_weight)
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
    else if (abs(eta - 1.) < 1e-10)
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
        return 1. / (0.5 + 0.25 * (1. - eta * eta) / eta * log((1 + eta)/abs(1 - eta)))
                 - 3. * vw_weight * eta * eta - tf_weight;
    }
}

void KEDF_ML::multiKernel(double *pinput, ModulePW::PW_Basis *pw_rho, double *routput)
{
    std::complex<double> *recipOutput = new std::complex<double>[pw_rho->npw];

    pw_rho->real2recip(pinput, recipOutput);
    for (int ip = 0; ip < pw_rho->npw; ++ip)
    {
        recipOutput[ip] *= this->kernel[ip];
    }
    pw_rho->recip2real(recipOutput, routput);

    delete[] recipOutput;
}