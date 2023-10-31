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
    std::string device_inpt,
    ModulePW::PW_Basis *pw_rho
)
{
    torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kDouble));
    auto output = torch::get_default_dtype();
    std::cout << "Default type: " << output << std::endl;

    this->set_device(device_inpt);

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
        this->nn = std::make_shared<NN_OFImpl>(this->nx, this->ninput, nnode, nlayer, this->device);
        torch::load(this->nn, "net.pt");
        this->nn->to(this->device);
        std::cout << "load net done" << std::endl;
        if (GlobalV::of_ml_feg != 0)
        {
            torch::Tensor feg_inpt = torch::zeros(this->ninput, this->device_type);
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
        this->ml_data->set_para(nx, nelec, tf_weight, vw_weight, 
                                chi_xi, chi_p, chi_q, chi_pnl, chi_qnl, pw_rho);
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
    this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/11_ks-pbe-chip0.2q0.1xi0.6-alpha2/1_fccAl-eq-2023-10-02/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/1_ks/2_bccAl_27dim-2022-12-12/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/3_ks-pbe-newpara/1_fccAl-eq-2023-02-14/rho.npy", temp_prho);
    // this->ml_data->loadVector("/home/dell/1_work/7_ABACUS_ML_OF/1_test/0_generate_data/5_ks-pbe-chip0.2q0.1/19_Li3Mg-mp-976254-eq-2023-03-20/rho.npy", temp_prho);
    double ** prho = new double *[1];
    prho[0] = new double[this->nx];
    for (int ir = 0; ir < this->nx; ++ir) prho[0][ir] = temp_prho[ir];
    std::cout << "Load rho done" << std::endl;
    // ==============================

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

    this->get_potential_(prho, pw_rho, potential);

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
    this->nn->zero_grad();
    this->nn->inputs.requires_grad_(false);
    this->nn->setData(this->nn_input_index, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl,
                      this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq, this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl,
                      this->device_type);
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
    ModuleBase::timer::tick("KEDF_ML", "updateInput");
}