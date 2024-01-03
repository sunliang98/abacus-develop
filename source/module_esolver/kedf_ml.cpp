#include "./kedf_ml.h"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/global_function.h"
// #include "time.h"

void KEDF_ML::set_para(
    const int nx, 
    const double dV, 
    const double nelec, 
    const double tf_weight, 
    const double vw_weight, 
    const double chi_p,
    const double chi_q,
    const std::string chi_xi_,
    const std::string chi_pnl_,
    const std::string chi_qnl_,
    const int nnode,
    const int nlayer,
    const int &nkernel,
    const std::string &kernel_type_,
    const std::string &kernel_scaling_,
    const std::string &yukawa_alpha_,
    const bool &of_ml_gamma,
    const bool &of_ml_p,
    const bool &of_ml_q,
    const bool &of_ml_tanhp,
    const bool &of_ml_tanhq,
    const std::string &of_ml_gammanl_,
    const std::string &of_ml_pnl_,
    const std::string &of_ml_qnl_,
    const std::string &of_ml_xi_,
    const std::string &of_ml_tanhxi_,
    const std::string &of_ml_tanhxi_nl_,
    const std::string &of_ml_tanh_pnl_,
    const std::string &of_ml_tanh_qnl_,
    const std::string &of_ml_tanhp_nl_,
    const std::string &of_ml_tanhq_nl_,
    const std::string device_inpt,
    ModulePW::PW_Basis *pw_rho
)
{
    torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kDouble));
    auto output = torch::get_default_dtype();
    std::cout << "Default type: " << output << std::endl;

    this->set_device(device_inpt);

    this->nx = nx;
    this->nx_tot = nx;
    this->dV = dV;
    this->nkernel = nkernel;
    this->chi_p = chi_p;
    this->chi_q = chi_q;
    this->ml_data->split_string(chi_xi_, nkernel, 1., this->chi_xi);
    this->ml_data->split_string(chi_pnl_, nkernel, 1., this->chi_pnl);
    this->ml_data->split_string(chi_qnl_, nkernel, 1., this->chi_qnl);

    this->init_data(
        nkernel,
        of_ml_gamma,
        of_ml_p,
        of_ml_q,
        of_ml_tanhp,
        of_ml_tanhq,
        of_ml_gammanl_,
        of_ml_pnl_,
        of_ml_qnl_,
        of_ml_xi_,
        of_ml_tanhxi_,
        of_ml_tanhxi_nl_,
        of_ml_tanh_pnl_,
        of_ml_tanh_qnl_,
        of_ml_tanhp_nl_,
        of_ml_tanhq_nl_);

    std::cout << "ninput = " << ninput << std::endl;

    if (GlobalV::of_kinetic == "ml")
    {
        this->nn = std::make_shared<NN_OFImpl>(this->nx, 0, this->ninput, nnode, nlayer, this->device);
        torch::load(this->nn, "net.pt", this->device_type);
        std::cout << "load net done" << std::endl;
        if (GlobalV::of_ml_feg != 0)
        {
            torch::Tensor feg_inpt = torch::zeros(this->ninput, this->device_type);
            for (int i = 0; i < this->ninput; ++i)
            {
                if (this->descriptor_type[i] == "gamma") feg_inpt[i] = 1.;
            }

            // feg_inpt.requires_grad_(true);

            if (GlobalV::of_ml_feg == 1) 
                // this->feg_net_F = torch::softplus(this->nn->forward(feg_inpt));
                this->feg_net_F = torch::softplus(this->nn->forward(feg_inpt)).to(this->device_CPU).contiguous().data_ptr<double>()[0];
            else
            {
                // this->feg_net_F = this->nn->forward(feg_inpt);
                this->feg_net_F = this->nn->forward(feg_inpt).to(this->device_CPU).contiguous().data_ptr<double>()[0];
            }

            std::cout << "feg_net_F = " << this->feg_net_F << std::endl;
        }
    } 
    
    if (GlobalV::of_kinetic == "ml" || GlobalV::of_ml_gene_data == 1)
    {
        this->ml_data->set_para(nx, nelec, tf_weight, vw_weight, chi_p, chi_q,
                                chi_xi_, chi_pnl_, chi_qnl_, nkernel, kernel_type_, kernel_scaling_, yukawa_alpha_, pw_rho);
    }
}

double KEDF_ML::get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    this->updateInput(prho, pw_rho);

    this->NN_forward(prho, pw_rho, false);
    
    torch::Tensor enhancement_cpu_tensor = this->nn->F.to(this->device_CPU).contiguous();
    this->enhancement_cpu_ptr = enhancement_cpu_tensor.data_ptr<double>();

    double energy = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        energy += enhancement_cpu_ptr[ir] * pow(prho[0][ir], 5./3.);
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

    this->NN_forward(prho, pw_rho, true);
    
    torch::Tensor enhancement_cpu_tensor = this->nn->F.to(this->device_CPU).contiguous();
    this->enhancement_cpu_ptr = enhancement_cpu_tensor.data_ptr<double>();
    torch::Tensor gradient_cpu_tensor = this->nn->inputs.grad().to(this->device_CPU).contiguous();
    this->gradient_cpu_ptr = gradient_cpu_tensor.data_ptr<double>();
    // std::cout << "F" << torch::slice(this->nn->F, 0, 0, 10) << std::endl;
    // this->enhancement_cpu_ptr = this->nn->F.to(this->device_CPU).contiguous().data_ptr<double>();
    // std::cout << "F_CPU" << torch::slice(this->nn->F.to(this->device_CPU), 0, 0, 10) << std::endl;
    // std::cout << "F_CPU_cont" << torch::slice(enhancement_cpu_tensor, 0, 0, 10) << std::endl;
    // std::cout << "enhancement_cpu_ptr" << std::endl;
    // for (int i = 0; i < 10; ++i) std::cout << enhancement_cpu_ptr[i] << "\t";
    // std::cout << std::endl;

    this->get_potential_(prho, pw_rho, rpotential);

    // get energy
    ModuleBase::timer::tick("KEDF_ML", "Pauli Energy");
    double energy = 0.;
    for (int ir = 0; ir < this->nx; ++ir)
    {
        // energy += this->nn->F[ir].item<double>() * pow(prho[0][ir], 5./3.);
        energy += enhancement_cpu_ptr[ir] * pow(prho[0][ir], 5./3.);
    }
    energy *= this->dV * this->cTF;
    this->MLenergy = energy;
    Parallel_Reduce::reduce_double_all(this->MLenergy);
    ModuleBase::timer::tick("KEDF_ML", "Pauli Energy");
}

void KEDF_ML::generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf,  ModulePW::PW_Basis *pw_rho, const double *veff)
{
    this->ml_data->generateTrainData_WT(prho, wt, tf, pw_rho, veff);
    if (GlobalV::of_kinetic == "ml")
    {
        this->updateInput(prho, pw_rho);

        this->NN_forward(prho, pw_rho, true);
        
        torch::Tensor enhancement_cpu_tensor = this->nn->F.to(this->device_CPU).contiguous();
        this->enhancement_cpu_ptr = enhancement_cpu_tensor.data_ptr<double>();
        torch::Tensor gradient_cpu_tensor = this->nn->inputs.grad().to(this->device_CPU).contiguous();
        this->gradient_cpu_ptr = gradient_cpu_tensor.data_ptr<double>();

        // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
        // std::cout << torch::slice(this->nn->F, 0, 0, 10) << std::endl;

        torch::Tensor enhancement = this->nn->F.reshape({this->nx});
        ModuleBase::matrix potential(1, this->nx);
        // torch::Tensor potential = torch::zeros_like(enhancement);

        this->get_potential_(prho, pw_rho, potential);

        std::cout << "dumpdump\n";
        this->dumpTensor(enhancement, "enhancement.npy");
        this->dumpMatrix(potential, "potential.npy");
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
    this->ml_data->loadVector("/home/xianyuer/data/1_sunliang/1_work/0_ml_kedf/1_test/0_generate_data/5_ks-pbe-chip0.2q0.1/1_fccAl-eq-2023-03-20/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/1_ks/2_bccAl_27dim-2022-12-12/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/3_ks-pbe-newpara/1_fccAl-eq-2023-02-14/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/5_ks-pbe-chip0.2q0.1/19_Li3Mg-mp-976254-eq-2023-03-20/rho.npy", temp_prho);
    double ** prho = new double *[1];
    prho[0] = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir) prho[0][ir] = temp_prho[ir];
    std::cout << "Load rho done" << std::endl;
    // ==============================

    this->updateInput(prho, pw_rho);
    std::cout << "update done" << std::endl;

    this->NN_forward(prho, pw_rho, true);
    
    torch::Tensor enhancement_cpu_tensor = this->nn->F.to(this->device_CPU).contiguous();
    this->enhancement_cpu_ptr = enhancement_cpu_tensor.data_ptr<double>();
    torch::Tensor gradient_cpu_tensor = this->nn->inputs.grad().to(this->device_CPU).contiguous();
    this->gradient_cpu_ptr = gradient_cpu_tensor.data_ptr<double>();

    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
    // std::cout << torch::slice(this->nn->F, 0, 0, 10) << std::endl;
    std::cout << "enhancement done" << std::endl;

    torch::Tensor enhancement = this->nn->F.reshape({this->nx});
    ModuleBase::matrix potential(1, this->nx);

    this->get_potential_(prho, pw_rho, potential);
    std::cout << "potential done" << std::endl;

    this->dumpTensor(enhancement, "enhancement-abacus.npy");
    this->dumpMatrix(potential, "potential-abacus.npy");
    exit(0);
}

void KEDF_ML::set_device(std::string device_inpt)
{
    if (device_inpt == "cpu")
    {
        std::cout << "---------- Running NN on CPU ----------" << std::endl;
        this->device_type = torch::kCPU;
    }
    else if (device_inpt == "gpu")
    {
        if (torch::cuda::cudnn_is_available())
        {
            std::cout << "---------- Running NN on GPU ----------" << std::endl;
            this->device_type = torch::kCUDA;
        }
        else
        {
            std::cout << "------ Warning: GPU is unaviable ------" << std::endl;
            std::cout << "---------- Running NN on CPU ----------" << std::endl;
            this->device_type = torch::kCPU;
        }
    }
    this->device = torch::Device(this->device_type);
}

void KEDF_ML::NN_forward(const double * const * prho, ModulePW::PW_Basis *pw_rho, bool cal_grad)
{
    ModuleBase::timer::tick("KEDF_ML", "Forward");
    // std::cout << "nn_forward" << std::endl;

    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    // this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
    //                   this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl,
    //                   this->device_type);
    this->nn->set_data(this, this->descriptor_type, this->kernel_index, this->nn->inputs);
    this->nn->inputs.requires_grad_(true);

    this->nn->F = this->nn->forward(this->nn->inputs);    
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.

    if (GlobalV::of_ml_feg != 3)
    {
        this->nn->F = torch::softplus(this->nn->F);
    }
    if (GlobalV::of_ml_feg == 1)
    {
        this->nn->F = this->nn->F - this->feg_net_F + 1.;
    }
    else if (GlobalV::of_ml_feg == 3)
    {
        this->nn->F = torch::softplus(this->nn->F - this->feg_net_F + this->feg3_correct);
    }
    ModuleBase::timer::tick("KEDF_ML", "Forward");

    if (cal_grad)
    {
        ModuleBase::timer::tick("KEDF_ML", "Backward");
        this->nn->F.backward(torch::ones({this->nx, 1}, this->device_type));
        ModuleBase::timer::tick("KEDF_ML", "Backward");
    }
    // std::cout << "nn_forward" << std::endl;
}

void KEDF_ML::dumpTensor(const torch::Tensor &data, std::string filename)
{
    std::cout << "Dumping " << filename << std::endl;
    torch::Tensor data_cpu = data.to(this->device_CPU).contiguous();
    std::vector<double> v(data_cpu.data_ptr<double>(), data_cpu.data_ptr<double>() + data_cpu.numel());
    // for (int ir = 0; ir < this->nx; ++ir) assert(v[ir] == data[ir].item<double>());
    this->ml_data->dumpVector(filename, v);
}

void KEDF_ML::dumpMatrix(const ModuleBase::matrix &data, std::string filename)
{
    std::cout << "Dumping " << filename << std::endl;
    std::vector<double> v(data.c, data.c + this->nx);
    // for (int ir = 0; ir < this->nx; ++ir) assert(v[ir] == data[ir].item<double>());
    this->ml_data->dumpVector(filename, v);
}

void KEDF_ML::updateInput(const double * const * prho, ModulePW::PW_Basis *pw_rho)
{
    ModuleBase::timer::tick("KEDF_ML", "updateInput");
    // std::cout << "updata_input" << std::endl;
    if (this->gene_data_label["gamma"][0])   this->ml_data->getGamma(prho, this->gamma);
    if (this->gene_data_label["p"][0])
    {
        this->ml_data->getNablaRho(prho, pw_rho, this->nablaRho);
        this->ml_data->getP(prho, pw_rho, this->nablaRho, this->p);
    }
    if (this->gene_data_label["q"][0])       this->ml_data->getQ(prho, pw_rho, this->q);
    if (this->gene_data_label["tanhp"][0])   this->ml_data->getTanhP(this->p, this->tanhp);
    if (this->gene_data_label["tanhq"][0])   this->ml_data->getTanhQ(this->q, this->tanhq);

    for (int ik = 0; ik < nkernel; ++ik)
    {
        if (this->gene_data_label["gammanl"][ik]){
            this->ml_data->getGammanl(ik, this->gamma, pw_rho, this->gammanl[ik]);
        }
        if (this->gene_data_label["pnl"][ik]){
            this->ml_data->getPnl(ik, this->p, pw_rho, this->pnl[ik]);
        }
        if (this->gene_data_label["qnl"][ik]){
            this->ml_data->getQnl(ik, this->q, pw_rho, this->qnl[ik]);
        }
        if (this->gene_data_label["xi"][ik]){
            this->ml_data->getXi(this->gamma, this->gammanl[ik], this->xi[ik]);
        }
        if (this->gene_data_label["tanhxi"][ik]){
            this->ml_data->getTanhXi(ik, this->gamma, this->gammanl[ik], this->tanhxi[ik]);
        }
        if (this->gene_data_label["tanhxi_nl"][ik]){
            this->ml_data->getTanhXi_nl(ik, this->tanhxi[ik], pw_rho, this->tanhxi_nl[ik]);
        }
        if (this->gene_data_label["tanh_pnl"][ik]){
            this->ml_data->getTanh_Pnl(ik, this->pnl[ik], this->tanh_pnl[ik]);
        }
        if (this->gene_data_label["tanh_qnl"][ik]){
            this->ml_data->getTanh_Qnl(ik, this->qnl[ik], this->tanh_qnl[ik]);
        }
        if (this->gene_data_label["tanhp_nl"][ik]){
            this->ml_data->getTanhP_nl(ik, this->tanhp, pw_rho, this->tanhp_nl[ik]);
        }
        if (this->gene_data_label["tanhq_nl"][ik]){
            this->ml_data->getTanhQ_nl(ik, this->tanhq, pw_rho, this->tanhq_nl[ik]);
        }
    }
    ModuleBase::timer::tick("KEDF_ML", "updateInput");
}

torch::Tensor KEDF_ML::get_data(std::string parameter, const int ikernel){
    if (parameter == "gamma")       return torch::tensor(this->gamma, this->device_type);
    if (parameter == "p")           return torch::tensor(this->p, this->device_type);
    if (parameter == "q")           return torch::tensor(this->q, this->device_type);
    if (parameter == "tanhp")       return torch::tensor(this->tanhp, this->device_type);
    if (parameter == "tanhq")       return torch::tensor(this->tanhq, this->device_type);
    if (parameter == "gammanl")     return torch::tensor(this->gammanl[ikernel], this->device_type);
    if (parameter == "pnl")         return torch::tensor(this->pnl[ikernel], this->device_type);
    if (parameter == "qnl")         return torch::tensor(this->qnl[ikernel], this->device_type);
    if (parameter == "xi")          return torch::tensor(this->xi[ikernel], this->device_type);
    if (parameter == "tanhxi")      return torch::tensor(this->tanhxi[ikernel], this->device_type);
    if (parameter == "tanhxi_nl")   return torch::tensor(this->tanhxi_nl[ikernel], this->device_type);
    if (parameter == "tanh_pnl")    return torch::tensor(this->tanh_pnl[ikernel], this->device_type);
    if (parameter == "tanh_qnl")    return torch::tensor(this->tanh_qnl[ikernel], this->device_type);
    if (parameter == "tanhp_nl")    return torch::tensor(this->tanhp_nl[ikernel], this->device_type);
    if (parameter == "tanhq_nl")    return torch::tensor(this->tanhq_nl[ikernel], this->device_type);
    return torch::zeros({});
}