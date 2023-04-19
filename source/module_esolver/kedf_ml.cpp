#include "./kedf_ml.h"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/global_function.h"
// #include "time.h"

void KEDF_ML::set_para(
    int nx, 
    double dV, 
    double nelec, 
    double tf_weight, 
    double vw_weight, 
    double chi_xi,
    double chi_p,
    double chi_q,
    double chi_pnl,
    double chi_qnl,
    int nnode,
    int nlayer,
    ModulePW::PW_Basis *pw_rho
)
{
    torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kDouble));
    auto output = torch::get_default_dtype();
    std::cout << "Default type: " << output << std::endl;

    this->nx = nx;
    this->dV = dV;
    this->chi_xi = chi_xi;
    this->chi_p = chi_p;
    this->chi_q = chi_q;
    this->chi_pnl = chi_pnl;
    this->chi_qnl = chi_qnl;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1},
                            {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}, 
                            {"xi", -1}, {"tanhxi", -1}, {"tanhxi_nl", -1},
                            {"tanhp", -1}, {"tanhq", -1}, 
                            {"tanh_pnl", -1}, {"tanh_qnl", -1}, 
                            {"tanhp_nl", -1}, {"tanhq_nl", -1}};

    this->ninput = 0;
    if (GlobalV::of_ml_gamma || GlobalV::of_ml_gammanl || GlobalV::of_ml_xi || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl){
        this->gamma = std::vector<double>(this->nx);
        if (GlobalV::of_ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    }    
    if (GlobalV::of_ml_p || GlobalV::of_ml_pnl || GlobalV::of_ml_tanhp || GlobalV::of_ml_tanh_pnl || GlobalV::of_ml_tanhp_nl){
        this->p = std::vector<double>(this->nx);
        this->nablaRho = std::vector<std::vector<double> >(3, std::vector<double>(this->nx, 0.));
        if (GlobalV::of_ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_q || GlobalV::of_ml_qnl || GlobalV::of_ml_tanhq || GlobalV::of_ml_tanh_qnl || GlobalV::of_ml_tanhq_nl){
        this->q = std::vector<double>(this->nx);
        if (GlobalV::of_ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_gammanl || GlobalV::of_ml_xi || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl){
        this->gammanl = std::vector<double>(this->nx);
        if (GlobalV::of_ml_gammanl)
        {
            this->nn_input_index["gammanl"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_pnl || GlobalV::of_ml_tanh_pnl){
        this->pnl = std::vector<double>(this->nx);
        if (GlobalV::of_ml_pnl)
        {
            this->nn_input_index["pnl"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_qnl || GlobalV::of_ml_tanh_qnl){
        this->qnl = std::vector<double>(this->nx);
        if (GlobalV::of_ml_qnl)
        {
            this->nn_input_index["qnl"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_xi || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl){
        this->xi = std::vector<double>(this->nx);
        if (GlobalV::of_ml_xi)
        {
            this->nn_input_index["xi"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl){
        this->tanhxi = std::vector<double>(this->nx); // we assume ONLY ONE of xi and tanhxi is used.
        if (GlobalV::of_ml_tanhxi)
        {
            this->nn_input_index["tanhxi"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_tanhxi_nl){
        this->tanhxi_nl = std::vector<double>(this->nx);
        this->nn_input_index["tanhxi_nl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_tanhp || GlobalV::of_ml_tanhp_nl){
        this->tanhp = std::vector<double>(this->nx);
        if (GlobalV::of_ml_tanhp)
        {
            this->nn_input_index["tanhp"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_tanhq || GlobalV::of_ml_tanhq_nl){
        this->tanhq = std::vector<double>(this->nx);
        if (GlobalV::of_ml_tanhq)
        {
            this->nn_input_index["tanhq"] = this->ninput;
            this->ninput++;
        }
    }
    if (GlobalV::of_ml_tanh_pnl){
        this->tanh_pnl = std::vector<double>(this->nx);
        this->nn_input_index["tanh_pnl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_tanh_qnl){
        this->tanh_qnl = std::vector<double>(this->nx);
        this->nn_input_index["tanh_qnl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_tanhp_nl){
        this->tanhp_nl = std::vector<double>(this->nx);
        this->nn_input_index["tanhp_nl"] = this->ninput;
        this->ninput++;
    }
    if (GlobalV::of_ml_tanhq_nl){
        this->tanhq_nl = std::vector<double>(this->nx);
        this->nn_input_index["tanhq_nl"] = this->ninput;
        this->ninput++;
    }

    if (GlobalV::of_kinetic == "ml")
    {
        this->nn = std::make_shared<NN_OFImpl>(this->nx, this->ninput, nnode, nlayer);
        torch::load(this->nn, "net.pt");
        if (GlobalV::of_ml_feg != 0)
        {
            torch::Tensor feg_inpt = torch::zeros(this->ninput);
            if (GlobalV::of_ml_gamma) feg_inpt[this->nn_input_index["gamma"]] = 1.;
            if (GlobalV::of_ml_p) feg_inpt[this->nn_input_index["p"]] = 0.;
            if (GlobalV::of_ml_q) feg_inpt[this->nn_input_index["q"]] = 0.;
            if (GlobalV::of_ml_gammanl) feg_inpt[this->nn_input_index["gammanl"]] = 0.;
            if (GlobalV::of_ml_pnl) feg_inpt[this->nn_input_index["pnl"]] = 0.;
            if (GlobalV::of_ml_qnl) feg_inpt[this->nn_input_index["qnl"]] = 0.;
            if (GlobalV::of_ml_xi) feg_inpt[this->nn_input_index["xi"]] = 0;
            if (GlobalV::of_ml_tanhxi) feg_inpt[this->nn_input_index["tanhxi"]] = 0;
            if (GlobalV::of_ml_tanhxi_nl) feg_inpt[this->nn_input_index["tanhxi_nl"]] = 0;
            if (GlobalV::of_ml_tanhp) feg_inpt[this->nn_input_index["tanhp"]] = 0;
            if (GlobalV::of_ml_tanhq) feg_inpt[this->nn_input_index["tanhq"]] = 0;
            if (GlobalV::of_ml_tanh_pnl) feg_inpt[this->nn_input_index["tanh_pnl"]] = 0;
            if (GlobalV::of_ml_tanh_qnl) feg_inpt[this->nn_input_index["tanh_qnl"]] = 0;
            if (GlobalV::of_ml_tanhp_nl) feg_inpt[this->nn_input_index["tanhp_nl"]] = 0;
            if (GlobalV::of_ml_tanhq_nl) feg_inpt[this->nn_input_index["tanhq_nl"]] = 0;

            // feg_inpt.requires_grad_(true);

            this->feg_net_F = this->nn->forward(feg_inpt).item<double>();
            std::cout << "feg_net_F = " << this->feg_net_F << std::endl;
        }
    } 
    
    if (GlobalV::of_kinetic == "ml" || GlobalV::of_ml_gene_data == 1)
    {
        this->ml_data->set_para(nx, nelec, tf_weight, vw_weight, 
                                chi_xi, chi_p, chi_q, chi_pnl, chi_qnl, pw_rho);
    }
}

double KEDF_ML::get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    this->updateInput(prho, pw_rho);
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
                      this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl);

    this->nn->F = this->nn->forward(this->nn->inputs);
    if (GlobalV::of_ml_feg == 1)
    {
        this->nn->F = this->nn->F - this->feg_net_F + 1.;
    }
    else if (GlobalV::of_ml_feg == 3)
    {
        this->nn->F = torch::softplus(this->nn->F - this->feg_net_F + this->feg3_correct);
    }

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
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
                      this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);
    // cout << this->nn->inputs.grad();
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.

    if (GlobalV::of_ml_feg == 1)
    {
        this->nn->F = this->nn->F - this->feg_net_F + 1.;
    }
    else if (GlobalV::of_ml_feg == 3)
    {
        this->nn->F = torch::softplus(this->nn->F - this->feg_net_F + this->feg3_correct);
    }

    // cout << "begin backward" << endl;
    this->nn->F.backward(torch::ones({this->nx, 1}));
    // cout << this->nn->inputs.grad();
    this->nn->gradient = this->nn->inputs.grad();

    // get potential
    // cout << "begin potential" << endl;
    std::vector<double> gammanlterm(this->nx, 0.);
    std::vector<double> xinlterm(this->nx, 0.);
    std::vector<double> tanhxinlterm(this->nx, 0.);
    std::vector<double> tanhxi_nlterm(this->nx, 0.);
    std::vector<double> ppnlterm(this->nx, 0.);
    std::vector<double> qqnlterm(this->nx, 0.);
    std::vector<double> tanhptanh_pnlterm(this->nx, 0.);
    std::vector<double> tanhqtanh_qnlterm(this->nx, 0.);
    std::vector<double> tanhptanhp_nlterm(this->nx, 0.);
    std::vector<double> tanhqtanhq_nlterm(this->nx, 0.);

    this->potGammanlTerm(prho, pw_rho, gammanlterm);
    this->potXinlTerm(prho, pw_rho, xinlterm);
    this->potTanhxinlTerm(prho, pw_rho, tanhxinlterm);
    this->potTanhxi_nlTerm(prho, pw_rho, tanhxi_nlterm);
    this->potPPnlTerm(prho, pw_rho, ppnlterm);
    this->potQQnlTerm(prho, pw_rho, qqnlterm);
    this->potTanhpTanh_pnlTerm(prho, pw_rho, tanhptanh_pnlterm);
    this->potTanhqTanh_qnlTerm(prho, pw_rho, tanhqtanh_qnlterm);
    this->potTanhpTanhp_nlTerm(prho, pw_rho, tanhptanhp_nlterm);
    this->potTanhqTanhq_nlTerm(prho, pw_rho, tanhqtanhq_nlterm);

    double kinetic_pot = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        kinetic_pot = this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
                      (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir)
                      + this->potXiTerm1(ir) + this->potTanhxiTerm1(ir) + this->potTanhpTerm1(ir) + this->potTanhqTerm1(ir))
                      + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir]
                      + xinlterm[ir] + tanhxinlterm[ir] + tanhxi_nlterm[ir]
                      + tanhptanh_pnlterm[ir] + tanhqtanh_qnlterm[ir]
                      + tanhptanhp_nlterm[ir] + tanhqtanhq_nlterm[ir];
        rpotential(0, ir) += kinetic_pot;

        // if (this->nn->F[ir][0].item<double>() < 0)
        // {
        //     std::cout << "WARNING: enhancement factor < 0 !!  " << this->nn->F[ir][0].item<double>() << std::endl;
        // }
        // if (kinetic_pot < 0)
        // {
        //     std::cout << "WARNING: pauli potential < 0 !!  " << kinetic_pot << std::endl;
        // }
        // rpotential(0, ir) += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
        //                     (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir))
        //                     + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir];
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
    if (GlobalV::of_kinetic == "ml")
    {
        this->updateInput(prho, pw_rho);

        this->nn->zero_grad();
        this->nn->inputs.requires_grad_(false);
        this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
                        this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl);
        this->nn->inputs.requires_grad_(true);

        this->nn->F = this->nn->forward(this->nn->inputs);
        if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
        // start = clock();

        if (GlobalV::of_ml_feg == 1)
        {
            this->nn->F = this->nn->F - this->feg_net_F + 1.;
        }
        else if (GlobalV::of_ml_feg == 3)
        {
            this->nn->F = torch::softplus(this->nn->F - this->feg_net_F + this->feg3_correct);
        }
        
        this->nn->F.backward(torch::ones({this->nx, 1}));
        // end = clock();
        // std::cout << "spend " << (end-start)/1e6 << " s" << std::endl;
        this->nn->gradient = this->nn->inputs.grad();

        // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
        // std::cout << torch::slice(this->nn->F, 0, 0, 10) << std::endl;

        torch::Tensor enhancement = this->nn->F.reshape({this->nx});
        torch::Tensor potential = torch::zeros_like(enhancement);

        // get potential
        std::vector<double> gammanlterm(this->nx, 0.);
        std::vector<double> xinlterm(this->nx, 0.);
        std::vector<double> tanhxinlterm(this->nx, 0.);
        std::vector<double> tanhxi_nlterm(this->nx, 0.);
        std::vector<double> ppnlterm(this->nx, 0.);
        std::vector<double> qqnlterm(this->nx, 0.);
        std::vector<double> tanhptanh_pnlterm(this->nx, 0.);
        std::vector<double> tanhqtanh_qnlterm(this->nx, 0.);
        std::vector<double> tanhptanhp_nlterm(this->nx, 0.);
        std::vector<double> tanhqtanhq_nlterm(this->nx, 0.);

        this->potGammanlTerm(prho, pw_rho, gammanlterm);
        this->potXinlTerm(prho, pw_rho, xinlterm);
        this->potTanhxinlTerm(prho, pw_rho, tanhxinlterm);
        this->potTanhxi_nlTerm(prho, pw_rho, tanhxi_nlterm);
        this->potPPnlTerm(prho, pw_rho, ppnlterm);
        this->potQQnlTerm(prho, pw_rho, qqnlterm);
        this->potTanhpTanh_pnlTerm(prho, pw_rho, tanhptanh_pnlterm);
        this->potTanhqTanh_qnlTerm(prho, pw_rho, tanhqtanh_qnlterm);
        this->potTanhpTanhp_nlTerm(prho, pw_rho, tanhptanhp_nlterm);
        this->potTanhqTanhq_nlTerm(prho, pw_rho, tanhqtanhq_nlterm);

        // sum over
        for (int ir = 0; ir < this->nx; ++ir)
        {
            // potential[ir] += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
            //                     (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir))
            //                     + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir];
            potential[ir] += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
                            (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir)
                            + this->potXiTerm1(ir) + this->potTanhxiTerm1(ir) + this->potTanhpTerm1(ir) + this->potTanhqTerm1(ir))
                            + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir]
                            + xinlterm[ir] + tanhxinlterm[ir] + tanhxi_nlterm[ir]
                            + tanhptanh_pnlterm[ir] + tanhqtanh_qnlterm[ir]
                            + tanhptanhp_nlterm[ir] + tanhqtanhq_nlterm[ir];
        }
        this->dumpTensor(enhancement, "enhancement.npy");
        this->dumpTensor(potential, "potential.npy");
    }
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
    this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/3_ks-pbe-newpara/1_fccAl-eq-2023-02-14/rho.npy", temp_prho);
    double ** prho = new double *[1];
    prho[0] = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir) prho[0][ir] = temp_prho[ir];
    std::cout << "Load rho done" << std::endl;
    // ==============================

    this->updateInput(prho, pw_rho);

    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
                      this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // start = clock();

    if (GlobalV::of_ml_feg == 1)
    {
        this->nn->F = this->nn->F - this->feg_net_F + 1.;
    }
    else if (GlobalV::of_ml_feg == 3)
    {
        this->nn->F = torch::softplus(this->nn->F - this->feg_net_F + this->feg3_correct);
    }

    this->nn->F.backward(torch::ones({this->nx, 1}));
    // end = clock();
    // std::cout << "spend " << (end-start)/1e6 << " s" << std::endl;
    this->nn->gradient = this->nn->inputs.grad();

    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
    // std::cout << torch::slice(this->nn->F, 0, 0, 10) << std::endl;

    torch::Tensor enhancement = this->nn->F.reshape({this->nx});
    torch::Tensor potential = torch::zeros_like(enhancement);

    // get potential
    std::vector<double> gammanlterm(this->nx, 0.);
    std::vector<double> xinlterm(this->nx, 0.);
    std::vector<double> tanhxinlterm(this->nx, 0.);
    std::vector<double> tanhxi_nlterm(this->nx, 0.);
    std::vector<double> ppnlterm(this->nx, 0.);
    std::vector<double> qqnlterm(this->nx, 0.);
    std::vector<double> tanhptanh_pnlterm(this->nx, 0.);
    std::vector<double> tanhqtanh_qnlterm(this->nx, 0.);
    std::vector<double> tanhptanhp_nlterm(this->nx, 0.);
    std::vector<double> tanhqtanhq_nlterm(this->nx, 0.);

    this->potGammanlTerm(prho, pw_rho, gammanlterm);
    this->potXinlTerm(prho, pw_rho, xinlterm);
    this->potTanhxinlTerm(prho, pw_rho, tanhxinlterm);
    this->potTanhxi_nlTerm(prho, pw_rho, tanhxi_nlterm);
    this->potPPnlTerm(prho, pw_rho, ppnlterm);
    this->potQQnlTerm(prho, pw_rho, qqnlterm);
    this->potTanhpTanh_pnlTerm(prho, pw_rho, tanhptanh_pnlterm);
    this->potTanhqTanh_qnlTerm(prho, pw_rho, tanhqtanh_qnlterm);
    this->potTanhpTanhp_nlTerm(prho, pw_rho, tanhptanhp_nlterm);
    this->potTanhqTanhq_nlTerm(prho, pw_rho, tanhqtanhq_nlterm);

    // sum over
    for (int ir = 0; ir < this->nx; ++ir)
    {
        // potential[ir] += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
        //                     (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir))
        //                     + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir];
        potential[ir] += this->cTF * pow(prho[0][ir], 5./3.) / prho[0][ir] *
                         (5./3. * this->nn->F[ir].item<double>() + this->potGammaTerm(ir) + this->potPTerm1(ir) + this->potQTerm1(ir)
                        + this->potXiTerm1(ir) + this->potTanhxiTerm1(ir) + this->potTanhpTerm1(ir) + this->potTanhqTerm1(ir))
                        + ppnlterm[ir] + qqnlterm[ir] + gammanlterm[ir]
                        + xinlterm[ir] + tanhxinlterm[ir] + tanhxi_nlterm[ir]
                        + tanhptanh_pnlterm[ir] + tanhqtanh_qnlterm[ir]
                        + tanhptanhp_nlterm[ir] + tanhqtanhq_nlterm[ir];
    }
    this->dumpTensor(enhancement, "enhancement-abacus.npy");
    this->dumpTensor(potential, "potential-abacus.npy");
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

double KEDF_ML::potXiTerm1(int ir)
{
    return (GlobalV::of_ml_xi) ? -1./3. * xi[ir] * this->nn->gradient[ir][this->nn_input_index["xi"]].item<double>() : 0.;
}

double KEDF_ML::potTanhxiTerm1(int ir)
{
    return (GlobalV::of_ml_tanhxi) ? -1./3. * xi[ir] * this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi)
                                    * this->nn->gradient[ir][this->nn_input_index["tanhxi"]].item<double>() : 0.;
}

double KEDF_ML::potTanhpTerm1(int ir)
{
    return (GlobalV::of_ml_tanhp) ? - 8./3. * p[ir] * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                                 * this->nn->gradient[ir][this->nn_input_index["tanhp"]].item<double>() : 0.;
}

double KEDF_ML::potTanhqTerm1(int ir)
{
    return (GlobalV::of_ml_tanhq) ? - 5./3. * q[ir] * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                                 * this->nn->gradient[ir][this->nn_input_index["tanhq"]].item<double>() : 0.;
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

void KEDF_ML::potXinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rXinlTerm)
{
    if (!GlobalV::of_ml_xi) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 4./3.) * this->nn->gradient[ir][this->nn_input_index["xi"]].item<double>();
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rXinlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rXinlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
}

void KEDF_ML::potTanhxinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxinlTerm)
{
    if (!GlobalV::of_ml_tanhxi) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 4./3.) * this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi)
                    * this->nn->gradient[ir][this->nn_input_index["tanhxi"]].item<double>();
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rTanhxinlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rTanhxinlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
}

void KEDF_ML::potTanhxi_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxi_nlTerm)
{
    if (!GlobalV::of_ml_tanhxi_nl) return;
    double *dFdxi = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->nn->gradient[ir][this->nn_input_index["tanhxi_nl"]].item<double>();
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, dFdxi);
    for (int ir = 0; ir < this->nx; ++ir)
    {
        dFdxi[ir] *= this->ml_data->dtanh(this->tanhxi[ir], this->chi_xi) / pow(prho[0][ir], 1./3.);
    }
    this->ml_data->multiKernel(dFdxi, pw_rho, rTanhxi_nlTerm.data());
    for (int ir = 0; ir < this->nx; ++ir)
    {
        rTanhxi_nlTerm[ir] += - dFdxi[ir] * this->xi[ir];
        rTanhxi_nlTerm[ir] *= 1./3. * pow(prho[0][ir], -2./3.);
    }
    delete[] dFdxi;
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

void KEDF_ML::potTanhpTanh_pnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanh_pnlTerm)
{
    if (!GlobalV::of_ml_tanhp && !GlobalV::of_ml_tanh_pnl) return;
    // Note we assume that tanhp_nl and tanh_pnl will NOT be used together.
    if (GlobalV::of_ml_tanhp_nl) return;
    // cout << "begin tanhp" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_tanh_pnl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->ml_data->dtanh(this->tanh_pnl[ir], this->chi_pnl)
                         * this->nn->gradient[ir][this->nn_input_index["tanh_pnl"]].item<double>();
        }
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_tanhp)? - 3./20. * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                           * this->nn->gradient[ir][this->nn_input_index["tanhp"]].item<double>() * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_tanh_pnl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->ml_data->divergence(tempP, pw_rho, rTanhpTanh_pnlTerm.data());

    if (GlobalV::of_ml_tanh_pnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhpTanh_pnlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potTanhqTanh_qnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanh_qnlTerm)
{
    if (!GlobalV::of_ml_tanhq && !GlobalV::of_ml_tanh_qnl) return;
    // Note we assume that tanhq_nl and tanh_qnl will NOT be used together.
    if (GlobalV::of_ml_tanhq_nl) return;
    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_tanh_qnl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.) * this->ml_data->dtanh(this->tanh_qnl[ir], this->chi_qnl)
                         * this->nn->gradient[ir][this->nn_input_index["tanh_qnl"]].item<double>();
        }
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_tanhq)? 3./40. * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                    * this->nn->gradient[ir][this->nn_input_index["tanhq"]].item<double>() * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_tanh_qnl)
        {
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rTanhqTanh_qnlTerm.data());

    if (GlobalV::of_ml_tanh_qnl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhqTanh_qnlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
        }
    }
    delete[] tempQ;
    delete[] dFdqnl;
}

// Note we assume that tanhp_nl and tanh_pnl will NOT be used together.
void KEDF_ML::potTanhpTanhp_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanhp_nlTerm)
{
    if (!GlobalV::of_ml_tanhp_nl) return;
    // cout << "begin tanhp" << endl;
    double *dFdpnl = nullptr;
    if (GlobalV::of_ml_tanhp_nl)
    {
        dFdpnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] = this->cTF * pow(prho[0][ir], 5./3.)
                         * this->nn->gradient[ir][this->nn_input_index["tanhp_nl"]].item<double>();
        }
        this->ml_data->multiKernel(dFdpnl, pw_rho, dFdpnl);
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdpnl[ir] *= this->ml_data->dtanh(this->tanhp[ir], this->chi_p);
        }
    }

    double ** tempP = new double*[3];
    for (int i = 0; i < 3; ++i)
    {
        tempP[i] = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            tempP[i][ir] = (GlobalV::of_ml_tanhp)? - 3./20. * this->ml_data->dtanh(this->tanhp[ir], this->chi_p)
                           * this->nn->gradient[ir][this->nn_input_index["tanhp"]].item<double>() * nablaRho[i][ir] / prho[0][ir] * /*Ha2Ry*/ 2. : 0.;
            if (GlobalV::of_ml_tanhp_nl)
            {
                tempP[i][ir] += - this->pqcoef * 2. * this->nablaRho[i][ir] / pow(prho[0][ir], 8./3.) * dFdpnl[ir];
            }
        }
    }
    this->ml_data->divergence(tempP, pw_rho, rTanhpTanhp_nlTerm.data());

    if (GlobalV::of_ml_tanhp_nl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhpTanhp_nlTerm[ir] += -8./3. * this->p[ir]/prho[0][ir] * dFdpnl[ir];
        }
    }

    for (int i = 0; i < 3; ++i) delete[] tempP[i];
    delete[] tempP;
    delete[] dFdpnl;
}

void KEDF_ML::potTanhqTanhq_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanhq_nlTerm)
{
    if (!GlobalV::of_ml_tanhq_nl) return;

    double *dFdqnl = nullptr;
    if (GlobalV::of_ml_tanhq_nl)
    {
        dFdqnl = new double[this->nx];
        for (int ir = 0; ir < this->nx; ++ir)
        {
            dFdqnl[ir] = this->cTF * pow(prho[0][ir], 5./3.)
                         * this->nn->gradient[ir][this->nn_input_index["tanhq_nl"]].item<double>();
        }
        this->ml_data->multiKernel(dFdqnl, pw_rho, dFdqnl);
    }

    double * tempQ = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir)
    {
        tempQ[ir] = (GlobalV::of_ml_tanhq)? 3./40. * this->ml_data->dtanh(this->tanhq[ir], this->chi_q)
                    * this->nn->gradient[ir][this->nn_input_index["tanhq"]].item<double>() * /*Ha2Ry*/ 2. : 0.;
        if (GlobalV::of_ml_tanhq_nl)
        {
            dFdqnl[ir] *= this->ml_data->dtanh(this->tanhq[ir], this->chi_q);
            tempQ[ir] += this->pqcoef / pow(prho[0][ir], 5./3.) * dFdqnl[ir];
        }
    }
    this->ml_data->Laplacian(tempQ, pw_rho, rTanhqTanhq_nlTerm.data());

    if (GlobalV::of_ml_tanhq_nl)
    {
        for (int ir = 0; ir < this->nx; ++ir)
        {
            rTanhqTanhq_nlTerm[ir] += -5./3. * this->q[ir] / prho[0][ir] * dFdqnl[ir];
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
    if (this->nn_input_index["gammanl"] >= 0 || this->nn_input_index["gamma"] >= 0
        || this->nn_input_index["xi"] >= 0 || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl)
    {
        this->ml_data->getGamma(prho, this->gamma);
        if (this->nn_input_index["gammanl"] >= 0 || this->nn_input_index["xi"] >= 0 || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl)
        {
            this->ml_data->getGammanl(this->gamma, pw_rho, this->gammanl);
            if (this->nn_input_index["xi"] >= 0 || GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl)
            {
                this->ml_data->getXi(this->gamma, this->gammanl, this->xi);
                if (GlobalV::of_ml_tanhxi || GlobalV::of_ml_tanhxi_nl)
                {
                    this->ml_data->tanh(this->xi, this->tanhxi, this->chi_xi);
                    if (GlobalV::of_ml_tanhxi_nl)
                    {
                        this->ml_data->getTanhXi_nl(this->tanhxi, pw_rho, this->tanhxi_nl);
                    }
                }
            }
            // if (GlobalV::of_ml_tanhxi)
            // {
            //     this->ml_data->getTanhXi(this->gamma, this->gammanl, this->tanhxi);
            // }
        }
    }
    if (this->nn_input_index["pnl"] >= 0 || this->nn_input_index["p"] >= 0
        || GlobalV::of_ml_tanhp || GlobalV::of_ml_tanhp_nl || GlobalV::of_ml_tanh_pnl)
    {
        this->ml_data->getNablaRho(prho, pw_rho, this->nablaRho);
        this->ml_data->getP(prho, pw_rho, this->nablaRho, this->p);
        if (this->nn_input_index["pnl"] >= 0 || GlobalV::of_ml_tanh_pnl)
        {
            this->ml_data->getPnl(this->p, pw_rho, this->pnl);
            if (GlobalV::of_ml_tanh_pnl)
            {
                this->ml_data->getTanh_Pnl(this->pnl, this->tanh_pnl);
            }
        }
        if (GlobalV::of_ml_tanhp || GlobalV::of_ml_tanhp_nl)
        {
            this->ml_data->getTanhP(this->p, this->tanhp);
            if (GlobalV::of_ml_tanhp_nl)
            {
                this->ml_data->getTanhP_nl(this->tanhp, pw_rho, this->tanhp_nl);
            }
        }
    }
    if (this->nn_input_index["qnl"] >= 0 || this->nn_input_index["q"] >= 0
        || GlobalV::of_ml_tanhq || GlobalV::of_ml_tanhq_nl || GlobalV::of_ml_tanh_qnl)
    {
        this->ml_data->getQ(prho, pw_rho, this->q);
        if (this->nn_input_index["qnl"] >= 0 || GlobalV::of_ml_tanh_qnl)
        {
            this->ml_data->getQnl(this->q, pw_rho, this->qnl);
            if (GlobalV::of_ml_tanh_qnl)
            {
                this->ml_data->getTanh_Qnl(this->qnl, this->tanh_qnl);
            }
        }
        if (GlobalV::of_ml_tanhq || GlobalV::of_ml_tanhq_nl)
        {
            this->ml_data->getTanhQ(this->q, this->tanhq);
            if (GlobalV::of_ml_tanhq_nl)
            {
                this->ml_data->getTanhQ_nl(this->tanhq, pw_rho, this->tanhq_nl);
            }
        }
    }
}