#include "./train.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

void Train::loadData()
{
    this->initData();
    this->loadData(this->train_dir, this->nx_train, this->ntrain,
                   this->rho, this->gamma, this->p, this->q,
                   this->gammanl, this->pnl, this->qnl, this->nablaRho,
                   this->xi, this->tanhxi, this->tanhxi_nl, this->tanhp, this->tanhq,
                   this->tanh_pnl, this->tanh_qnl, this->tanhp_nl, this->tanhq_nl,
                   this->enhancement, this->enhancement_mean, this->tau_mean, this->pauli, this->pauli_mean);
    std::cout << "enhancement mean: " << this->enhancement_mean << std::endl;
    std::cout << "exponent: " << this->exponent << std::endl;
    std::cout << "tau mean: " << this->tau_mean << std::endl;
    std::cout << "pauli potential mean: " << this->pauli_mean << std::endl;

    if (this->nvalidation > 0)
    {
        this->loadData(this->validation_dir, this->nx_vali, this->nvalidation,
                       this->rho_vali, this->gamma_vali, this->p_vali, this->q_vali,
                       this->gammanl_vali, this->pnl_vali, this->qnl_vali, this->nablaRho_vali,
                       this->xi_vali, this->tanhxi_vali, this->tanhxi_nl_vali, this->tanhp_vali, this->tanhq_vali,
                       this->tanh_pnl_vali, this->tanh_qnl_vali, this->tanhp_nl_vali, this->tanhq_nl_vali, 
                       this->enhancement_vali, this->enhancement_mean_vali, this->tau_mean_vali, this->pauli_vali, this->pauli_mean_vali);
        this->input_vali = torch::zeros({this->nx_vali, this->ninput});
        if (this->nn_input_index["gamma"] >= 0)     this->input_vali.index({"...", this->nn_input_index["gamma"]})      = gamma_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["p"] >= 0)         this->input_vali.index({"...", this->nn_input_index["p"]})          = p_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["q"] >= 0)         this->input_vali.index({"...", this->nn_input_index["q"]})          = q_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["gammanl"] >= 0)   this->input_vali.index({"...", this->nn_input_index["gammanl"]})    = gammanl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["pnl"] >= 0)       this->input_vali.index({"...", this->nn_input_index["pnl"]})        = pnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["qnl"] >= 0)       this->input_vali.index({"...", this->nn_input_index["qnl"]})        = qnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["xi"] >= 0)        this->input_vali.index({"...", this->nn_input_index["xi"]})         = xi_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhxi"] >= 0)    this->input_vali.index({"...", this->nn_input_index["tanhxi"]})     = tanhxi_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhxi_nl"] >= 0) this->input_vali.index({"...", this->nn_input_index["tanhxi_nl"]})  = tanhxi_nl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhp"] >= 0)     this->input_vali.index({"...", this->nn_input_index["tanhp"]})      = tanhp_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhq"] >= 0)     this->input_vali.index({"...", this->nn_input_index["tanhq"]})      = tanhq_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanh_pnl"] >= 0)  this->input_vali.index({"...", this->nn_input_index["tanh_pnl"]})   = tanh_pnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanh_qnl"] >= 0)  this->input_vali.index({"...", this->nn_input_index["tanh_qnl"]})   = tanh_qnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhp_nl"] >= 0)  this->input_vali.index({"...", this->nn_input_index["tanhp_nl"]})   = tanhp_nl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["tanhq_nl"] >= 0)  this->input_vali.index({"...", this->nn_input_index["tanhq_nl"]})   = tanhq_nl_vali.reshape({this->nx_vali}).clone();
    }
    std::cout << "Load data done" << std::endl;
}

void Train::initData()
{
    this->nx = pow(this->fftdim, 3);
    this->nx_train = this->nx * this->ntrain;
    this->nx_vali = this->nx * this->nvalidation;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1},
                            {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}, 
                            {"xi", -1}, {"tanhxi", -1}, {"tanhxi_nl", -1},
                            {"tanhp", -1}, {"tanhq", -1}, 
                            {"tanh_pnl", -1}, {"tanh_qnl", -1}, 
                            {"tanhp_nl", -1}, {"tanhq_nl", -1}};

    this->fft_grid_train = std::vector<std::vector<torch::Tensor>>(this->ntrain);
    this->fft_gg_train = std::vector<torch::Tensor>(this->ntrain);
    this->fft_kernel_train = std::vector<torch::Tensor>(this->ntrain);
    for (int i = 0; i < this->ntrain; ++i)
    {
        this->fft_grid_train[i] = std::vector<torch::Tensor>(3);
    }

    this->fft_grid_vali = std::vector<std::vector<torch::Tensor>>(this->nvalidation);
    this->fft_gg_vali = std::vector<torch::Tensor>(this->nvalidation);
    this->fft_kernel_vali = std::vector<torch::Tensor>(this->nvalidation);
    for (int i = 0; i < this->nvalidation; ++i)
    {
        this->fft_grid_vali[i] = std::vector<torch::Tensor>(3);
    }

    this->ninput = 0;

    this->rho = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->enhancement = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->enhancement_mean = torch::zeros(this->ntrain);
    this->tau_mean = torch::zeros(this->ntrain);
    this->pauli = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->pauli_mean = torch::zeros(this->ntrain);
    this->gamma = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->p = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->nablaRho = torch::zeros({this->ntrain, 3, this->fftdim, this->fftdim, this->fftdim});
    this->q = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->gammanl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->pnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->qnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->xi = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhxi = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhxi_nl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhp = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhq = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanh_pnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanh_qnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhp_nl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->tanhq_nl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    if (this->nvalidation > 0)
    {
        this->rho_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->enhancement_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->enhancement_mean_vali = torch::zeros(this->nvalidation);
        this->tau_mean_vali = torch::zeros(this->nvalidation);
        this->pauli_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->pauli_mean_vali = torch::zeros(this->nvalidation);
        this->gamma_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->p_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->nablaRho_vali = torch::zeros({this->nvalidation, 3, this->fftdim, this->fftdim, this->fftdim});
        this->q_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->gammanl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->pnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->qnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->xi_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhxi_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhxi_nl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhp_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhq_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanh_pnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanh_qnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhp_nl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->tanhq_nl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
    }

    // if (this->ml_gamma || this->ml_gammanl){
        if (this->ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    // }    
    // if (this->ml_p || this->ml_pnl){
        if (this->ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    // }
    // if (this->ml_q || this->ml_qnl){
        if (this->ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    // }
    if (this->ml_gammanl){
        this->nn_input_index["gammanl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_pnl){
        this->nn_input_index["pnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_qnl){
        this->nn_input_index["qnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_xi){
        this->nn_input_index["xi"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhxi){
        this->nn_input_index["tanhxi"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhxi_nl){
        this->nn_input_index["tanhxi_nl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhp){
        this->nn_input_index["tanhp"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhq){
        this->nn_input_index["tanhq"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanh_pnl){
        this->nn_input_index["tanh_pnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanh_qnl){
        this->nn_input_index["tanh_qnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhp_nl){
        this->nn_input_index["tanhp_nl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_tanhq_nl){
        this->nn_input_index["tanhq_nl"] = this->ninput;
        this->ninput++;
    }

    std::cout << "ninput = " << this->ninput << std::endl;

    if (this->feg_limit != 0)
    {
        this->feg_inpt = torch::zeros(this->ninput);
        if (this->ml_gamma) {
            this->feg_inpt[this->nn_input_index["gamma"]] = 1.;
            // this->feg_inpt[this->nn_input_index["gamma"]].requires_grad_(true);
        }
        if (this->ml_p) this->feg_inpt[this->nn_input_index["p"]] = 0.;
        if (this->ml_q) this->feg_inpt[this->nn_input_index["q"]] = 0.;
        if (this->ml_gammanl) this->feg_inpt[this->nn_input_index["gammanl"]] = 0.;
        if (this->ml_pnl) this->feg_inpt[this->nn_input_index["pnl"]] = 0.;
        if (this->ml_qnl) this->feg_inpt[this->nn_input_index["qnl"]] = 0.;
        if (this->ml_xi)    this->feg_inpt[this->nn_input_index["xi"]] = 0.;
        if (this->ml_tanhxi)    this->feg_inpt[this->nn_input_index["tanhxi"]] = 0.;
        if (this->ml_tanhxi_nl) this->feg_inpt[this->nn_input_index["tanhxi_nl"]] = 0.;
        if (this->ml_tanhp) this->feg_inpt[this->nn_input_index["tanhp"]] = 0.;
        if (this->ml_tanhq) this->feg_inpt[this->nn_input_index["tanhq"]] = 0.;
        if (this->ml_tanh_pnl)  this->feg_inpt[this->nn_input_index["tanh_pnl"]] = 0.;
        if (this->ml_tanh_qnl)  this->feg_inpt[this->nn_input_index["tanh_qnl"]] = 0.;
        if (this->ml_tanhp_nl)  this->feg_inpt[this->nn_input_index["tanhp_nl"]] = 0.;
        if (this->ml_tanhq_nl)  this->feg_inpt[this->nn_input_index["tanhq_nl"]] = 0.;
        this->feg_inpt.requires_grad_(true);

        this->feg_predict = torch::zeros(1);
        this->feg_dFdgamma = torch::zeros(1);
    }
    std::cout << "feg_limit = " << this->feg_limit << std::endl;
}

void Train::loadData(
    std::string *dir, 
    int nx,
    int nDataSet,
    torch::Tensor &rho,
    torch::Tensor &gamma,
    torch::Tensor &p,
    torch::Tensor &q,
    torch::Tensor &gammanl,
    torch::Tensor &pnl,
    torch::Tensor &qnl,
    torch::Tensor &nablaRho,
    torch::Tensor &xi,
    torch::Tensor &tanhxi,
    torch::Tensor &tanhxi_nl,
    torch::Tensor &tanhp,
    torch::Tensor &tanhq,
    torch::Tensor &tanh_pnl,
    torch::Tensor &tanh_qnl,
    torch::Tensor &tanhp_nl,
    torch::Tensor &tanhq_nl,
    torch::Tensor &enhancement,
    torch::Tensor &enhancement_mean,
    torch::Tensor &tau_mean,
    torch::Tensor &pauli,
    torch::Tensor &pauli_mean
)
{
    if (nDataSet <= 0) return;
    
    std::vector<long unsigned int> cshape = {(long unsigned) nx};
    std::vector<double> container(nx);
    bool fortran_order = false;

    for (int idata = 0; idata < nDataSet; ++idata)
    {
        this->loadTensor(dir[idata] + "/rho.npy", cshape, fortran_order, container, idata, rho);
        if (this->ml_gamma || this->ml_gammanl || this->ml_xi || this->ml_tanhxi || this->ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/gamma.npy", cshape, fortran_order, container, idata, gamma);
        }
        if (this->ml_gammanl || this->ml_xi || this->ml_tanhxi || this->ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/gammanl.npy", cshape, fortran_order, container, idata, gammanl);
        }
        if (this->ml_p || this->ml_pnl || this->ml_tanhp || this->ml_tanh_pnl || this->ml_tanhp_nl)
        {
            this->loadTensor(dir[idata] + "/p.npy", cshape, fortran_order, container, idata, p);
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhox.npy", cshape, fortran_order, container);
            nablaRho[idata][0] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoy.npy", cshape, fortran_order, container);
            nablaRho[idata][1] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoz.npy", cshape, fortran_order, container);
            nablaRho[idata][2] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
        }
        if (this->ml_pnl || this->ml_tanh_pnl){
            this->loadTensor(dir[idata] + "/pnl.npy", cshape, fortran_order, container, idata, pnl);
        }
        if (this->ml_q || this->ml_qnl || this->ml_tanhq || this->ml_tanh_qnl || this->ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/q.npy", cshape, fortran_order, container, idata, q);
        }
        if (this->ml_qnl || this->ml_tanh_qnl){
            this->loadTensor(dir[idata] + "/qnl.npy", cshape, fortran_order, container, idata, qnl);
        }
        if (this->ml_xi || this->ml_tanhxi || this->ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/xi.npy", cshape, fortran_order, container, idata, xi);
        }
        if (this->ml_tanhxi || this->ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/tanhxi.npy", cshape, fortran_order, container, idata, tanhxi);
        }
        if (this->ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/tanhxi_nl.npy", cshape, fortran_order, container, idata, tanhxi_nl);
        }
        if (this->ml_tanhp || this->ml_tanhp_nl){
            this->loadTensor(dir[idata] + "/tanhp.npy", cshape, fortran_order, container, idata, tanhp);
        }
        if (this->ml_tanhq || this->ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/tanhq.npy", cshape, fortran_order, container, idata, tanhq);
        }
        if (this->ml_tanh_pnl){
            this->loadTensor(dir[idata] + "/tanh_pnl.npy", cshape, fortran_order, container, idata, tanh_pnl);
        }
        if (this->ml_tanh_qnl){
            this->loadTensor(dir[idata] + "/tanh_qnl.npy", cshape, fortran_order, container, idata, tanh_qnl);
        }
        if (this->ml_tanhp_nl){
            this->loadTensor(dir[idata] + "/tanhp_nl.npy", cshape, fortran_order, container, idata, tanhp_nl);
        }
        if (this->ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/tanhq_nl.npy", cshape, fortran_order, container, idata, tanhq_nl);
        }

        this->loadTensor(dir[idata] + "/enhancement.npy", cshape, fortran_order, container, idata, enhancement);
        enhancement_mean[idata] = torch::mean(enhancement[idata]);
        tau_mean[idata] = torch::mean(torch::pow(rho[idata], this->exponent/3.) * enhancement[idata]);

        if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new")
        {
            this->loadTensor(dir[idata] + "/pauli.npy", cshape, fortran_order, container, idata, pauli);
            pauli_mean[idata] = torch::mean(pauli[idata]);
        }
    }
    enhancement.resize_({nx, 1});
    pauli.resize_({nx, 1});
}

void Train::loadTensor(
    std::string file,
    std::vector<long unsigned int> cshape,
    bool fortran_order, 
    std::vector<double> &container,
    int index,
    torch::Tensor &data
)
{
    npy::LoadArrayFromNumpy(file, cshape, fortran_order, container);
    data[index] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
}

void Train::dumpTensor(const torch::Tensor &data, std::string filename, int nx)
{
    std::vector<double> v(nx);
    for (int ir = 0; ir < nx; ++ir) v[ir] = data[ir].item<double>();
    // std::vector<double> v(data.data_ptr<float>(), data.data_ptr<float>() + data.numel()); // this works, but only supports float tensor
    const long unsigned cshape[] = {(long unsigned) nx}; // shape
    npy::SaveArrayAsNumpy(filename, false, 1, cshape, v);
    std::cout << "Dumping " << filename << " done" << std::endl;
}
