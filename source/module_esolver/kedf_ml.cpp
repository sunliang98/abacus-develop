#include "./kedf_ml.h"
#include "npy.hpp"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/global_function.h"

void KEDF_ML::set_para(int nx, double dV, double nelec, double tf_weight, double vw_weight, ModulePW::PW_Basis *pw_rho)
{
    this->nx = nx;
    this->dV = dV;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1}, {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}};

    this->ninput = 0;
    if (GlobalV::of_ml_gamma || GlobalV::of_ml_gammanl){
        this->gamma = std::vector<double>(this->nx);
        if (GlobalV::of_ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    }    
    if (GlobalV::of_ml_p || GlobalV::of_ml_pnl){
        this->p = std::vector<double>(this->nx);
        this->nablaRho = std::vector<std::vector<double> >(3, std::vector<double>(this->nx, 0.));
        if (GlobalV::of_ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_q || GlobalV::of_ml_qnl){
        this->q = std::vector<double>(this->nx);
        if (GlobalV::of_ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_gammanl){
        this->gammanl = std::vector<double>(this->nx);
        this->nn_input_index["gammanl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_pnl){
        this->pnl = std::vector<double>(this->nx);
        this->nn_input_index["pnl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_qnl){
        this->qnl = std::vector<double>(this->nx);
        this->nn_input_index["qnl"] = this->ninput;
        this->ninput++;
    }

    if (GlobalV::of_kinetic == "ml")
    {
        this->nn = std::make_shared<NN_OFImpl>(this->nx, this->ninput);
        torch::load(this->nn, "net.pt");
    } 

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

double KEDF_ML::get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    this->updateInput(prho, pw_rho);
    this->nn->setData(this->nn_input_index, this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);

    this->nn->F = this->nn->forward(this->nn->inputs);

    double energy = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        energy += this->nn->F[ir][0].item<double>() * pow(prho[0][ir], 5./3.);
    }
    cout << "energy" << energy << endl;
    energy *= this->dV * this->cTF;
    this->MLenergy = energy;
    Parallel_Reduce::reduce_double_all(this->MLenergy);
    return this->MLenergy;
}

void KEDF_ML::ML_potential(const double * const * prho, ModulePW::PW_Basis *pw_rho, ModuleBase::matrix &rpotential)
{
    this->updateInput(prho, pw_rho);

    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    this->nn->setData(this->nn_input_index, this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);
    // cout << this->nn->inputs.grad();
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // cout << "begin backward" << endl;
    this->nn->F.backward(torch::ones({this->nx, 1}));
    // cout << this->nn->inputs.grad();
    this->nn->gradient = this->nn->inputs.grad();

    // get potential
    // cout << "begin potential" << endl;
    std::vector<double> gammanlterm(this->nx, 0.);
    std::vector<double> ppnlterm(this->nx, 0.);
    std::vector<double> qqnlterm(this->nx, 0.);

    this->potGammanlTerm(prho, pw_rho, gammanlterm);
    this->potPPnlTerm(prho, pw_rho, ppnlterm);
    this->potQQnlTerm(prho, pw_rho, qqnlterm);

    for (int ir = 0; ir < this->nx; ++ir)
    {
        rpotential(0, ir) += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
                            (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir))
                            + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir];
    }

    // get energy
    double energy = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        energy += this->nn->F[ir][0].item<double>() * pow(prho[0][ir], 5./3.);
    }
    energy *= this->dV * this->cTF;
    this->MLenergy = energy;
    Parallel_Reduce::reduce_double_all(this->MLenergy);
}

void KEDF_ML::generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf,  ModulePW::PW_Basis *pw_rho)
{
    // container which will contain gamma, p, q in turn
    std::vector<double> container(this->nx);
    // container contains gammanl, pnl, qnl in turn
    std::vector<double> containernl(this->nx);
    // nabla rho
    std::vector<std::vector<double> > nablaRho(3, std::vector<double>(this->nx, 0.));

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

void KEDF_ML::localTest(const double * const *prho, ModulePW::PW_Basis *pw_rho)
{
    this->updateInput(prho, pw_rho);

    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    this->nn->setData(this->nn_input_index, this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    this->nn->F.backward(torch::ones({this->nx, 1}));
    this->nn->gradient = this->nn->inputs.grad();

    torch::Tensor enhancement = this->nn->F.reshape({this->nx});
    torch::Tensor potential = torch::zeros_like(enhancement);

    // get potential
    std::vector<double> gammanlterm(this->nx, 0.);
    std::vector<double> ppnlterm(this->nx, 0.);
    std::vector<double> qqnlterm(this->nx, 0.);

    this->potGammanlTerm(prho, pw_rho, gammanlterm);
    this->potPPnlTerm(prho, pw_rho, ppnlterm);
    this->potQQnlTerm(prho, pw_rho, qqnlterm);

    // sum over
    for (int ir = 0; ir < this->nx; ++ir)
    {
        potential[ir] += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
                            (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir))
                            + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir];
    }
    this->dumpTensor(enhancement, "enhancement-test.npy");
    this->dumpTensor(potential, "potential-test.npy");
    exit(0);
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

double KEDF_ML::potGammaTerm(int ir)
{
    return (GlobalV::of_ml_gamma) ? 1./3. * gamma[ir] * this->nn->gradient[ir][this->nn_input_index["gamma"]].item<double>() : 0.;
}

double KEDF_ML::potPTerm1(int ir)
{
    return (GlobalV::of_ml_p) ? - 8./3. * p[ir] * this->nn->gradient[ir][this->nn_input_index["p"]].item<double>() : 0.;
}

double KEDF_ML::potQTerm1(int ir)
{
    return (GlobalV::of_ml_q) ? - 5./3. * q[ir] * this->nn->gradient[ir][this->nn_input_index["q"]].item<double>() : 0.;
}

void KEDF_ML::potGammanlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rGammanlTerm)
{
    if (!GlobalV::of_ml_gammanl) return;
    double *dFdgammanl = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdgammanl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->nn->gradient[ir][this->nn_input_index["gammanl"]].item<double>();
    }
    this->multiKernel(dFdgammanl, pw_rho, rGammanlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rGammanlTerm[ir] *= 1./3. * this->gamma[ir] / prho[0][ir];
    }
    delete[] dFdgammanl;
}

// get contribution of p and pnl
void KEDF_ML::potPPnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rPPnlTerm)
{
    if (!GlobalV::of_ml_p && !GlobalV::of_ml_pnl) return;
    // cout << "begin p" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_pnl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->nn->gradient[ir][this->nn_input_index["pnl"]].item<double>();
        }
        this->multiKernel(dFdpnl, pw_rho, dFdpnl);
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_p)? - 3./20. * this->nn->gradient[ir][this->nn_input_index["p"]].item<double>() * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_pnl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->divergence(tempP, pw_rho, rPPnlTerm.data());

    if (GlobalV::of_ml_pnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rPPnlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potQQnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rQQnlTerm)
{
    if (!GlobalV::of_ml_q && !GlobalV::of_ml_qnl) return;
    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_qnl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->nn->gradient[ir][this->nn_input_index["qnl"]].item<double>();
        }
        this->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_q)? 3./40. * this->nn->gradient[ir][this->nn_input_index["q"]].item<double>() * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_qnl)
        {
            tempQ[ir] += - this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->Laplacian(tempQ, pw_rho, rQQnlTerm.data());

    if (GlobalV::of_ml_qnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rQQnlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
        }
    }
    delete[] tempQ;
    delete[] dFdqnl;
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

void KEDF_ML::Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput)
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

void KEDF_ML::divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput)
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

void KEDF_ML::dumpTensor(const torch::Tensor &data, std::string filename)
{
    std::cout << "Dumping " << filename << std::endl;
    std::vector<double> v(this->nx);
    for (int ir = 0; ir < this->nx; ++ir) v[ir] = data[ir].item<double>();
    // std::vector<double> v(data.data_ptr<float>(), data.data_ptr<float>() + data.numel()); // this works, but only supports float tensor
    const long unsigned cshape[] = {(long unsigned) this->nx}; // shape
    npy::SaveArrayAsNumpy(filename, false, 1, cshape, v);
}

void KEDF_ML::updateInput(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    if (this->nn_input_index["gammanl"] >= 0 || this->nn_input_index["gamma"] >= 0)
    {
        this->getGamma(prho, this->gamma);
        if (this->nn_input_index["gammanl"] >= 0)
        {
            this->getGammanl(this->gamma, pw_rho, this->gammanl);
        }
    }
    if (this->nn_input_index["pnl"] >= 0 || this->nn_input_index["p"] >= 0)
    {
        this->getNablaRho(prho, pw_rho, this->nablaRho);
        this->getP(prho, pw_rho, this->nablaRho, this->p);
        if (this->nn_input_index["pnl"] >= 0)
        {
            this->getPnl(this->p, pw_rho, this->pnl);
        }
    }
    if (this->nn_input_index["qnl"] >= 0 || this->nn_input_index["q"] >= 0)
    {
        this->getQ(prho, pw_rho, this->q);
        if (this->nn_input_index["qnl"] >= 0)
        {
            this->getQnl(this->q, pw_rho, this->qnl);
        }
    }
}