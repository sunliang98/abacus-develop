#include "./kedf_ml.h"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/global_function.h"
// #include "time.h"

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
    
    if (GlobalV::of_kinetic == "ml" || GlobalV::of_ml_gene_data == 1)
    {
        this->ml_data->set_para(nx, nelec, tf_weight, vw_weight, pw_rho);
    }
}

double KEDF_ML::get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    this->updateInput(prho, pw_rho);
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl);

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
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl);
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

void KEDF_ML::generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf,  ModulePW::PW_Basis *pw_rho, const double *veff)
{
    this->ml_data->generateTrainData_WT(prho, wt, tf, pw_rho, veff);
}

void KEDF_ML::localTest(const double * const *pprho, ModulePW::PW_Basis *pw_rho)
{
    // time_t start, end;
    // for test =====================
    std::vector<long unsigned int> cshape = {(long unsigned) this->nx};
    bool fortran_order = false;

    std::vector<double> temp_prho(this->nx);
    // npy::LoadArrayFromNumpy("/home/dell/1_work/7_ABACUS_ML_OF/1_test/1_train/2022-11-11-potential-check/gpq/abacus/1_validation_set_bccAl/reference/rho.npy", cshape, fortran_order, temp_prho);
    // npy::LoadArrayFromNumpy("/home/dell/1_work/7_ABACUS_ML_OF/1_test/1_train/2022-11-11-potential-check/gpq/abacus/0_train_set/reference/rho.npy", cshape, fortran_order, temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/1_train/2022-11-11-potential-check/gpq/abacus/0_train_set/reference/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/1_ks/1_fccAl-2022-12-12/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/1_ks/2_bccAl_27dim-2022-12-12/rho.npy", temp_prho);
    this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/2_ks-pbe/1_fccAl-eq-2022-12-27/rho.npy", temp_prho);
    double ** prho = new double *[1];
    prho[0] = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir) prho[0][ir] = temp_prho[ir];
    std::cout << "Load rho done" << std::endl;
    // ==============================

    this->updateInput(prho, pw_rho);

    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // start = clock();
    this->nn->F.backward(torch::ones({this->nx, 1}));
    // end = clock();
    // std::cout << "spend " << (end-start)/1e6 << " s" << std::endl;
    this->nn->gradient = this->nn->inputs.grad();
    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;

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
    this->ml_data->multiKernel(dFdgammanl, pw_rho, rGammanlTerm.data());
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
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
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
    this->ml_data->divergence(tempP, pw_rho, rPPnlTerm.data());

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
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_q)? 3./40. * this->nn->gradient[ir][this->nn_input_index["q"]].item<double>() * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_qnl)
        {
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rQQnlTerm.data());

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

void KEDF_ML::dumpTensor(const torch::Tensor &data, std::string filename)
{
    std::cout << "Dumping " << filename << std::endl;
    std::vector<double> v(this->nx);
    for (int ir = 0; ir < this->nx; ++ir) v[ir] = data[ir].item<double>();
    this->ml_data->dumpVector(filename, v);
}

void KEDF_ML::updateInput(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    if (this->nn_input_index["gammanl"] >= 0 || this->nn_input_index["gamma"] >= 0)
    {
        this->ml_data->getGamma(prho, this->gamma);
        if (this->nn_input_index["gammanl"] >= 0)
        {
            this->ml_data->getGammanl(this->gamma, pw_rho, this->gammanl);
        }
    }
    if (this->nn_input_index["pnl"] >= 0 || this->nn_input_index["p"] >= 0)
    {
        this->ml_data->getNablaRho(prho, pw_rho, this->nablaRho);
        this->ml_data->getP(prho, pw_rho, this->nablaRho, this->p);
        if (this->nn_input_index["pnl"] >= 0)
        {
            this->ml_data->getPnl(this->p, pw_rho, this->pnl);
        }
    }
    if (this->nn_input_index["qnl"] >= 0 || this->nn_input_index["q"] >= 0)
    {
        this->ml_data->getQ(prho, pw_rho, this->q);
        if (this->nn_input_index["qnl"] >= 0)
        {
            this->ml_data->getQnl(this->q, pw_rho, this->qnl);
        }
    }
}