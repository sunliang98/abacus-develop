#include "./train.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

void Train::loadData()
{
    this->initData();
    this->loadData(this->train_dir, this->nx_train, this->ntrain, this->rho, this->gamma, this->p, this->q,
                   this->gammanl, this->pnl, this->qnl, this->nablaRho, this->enhancement, this->enhancement_mean, this->pauli, this->pauli_mean);
    std::cout << "enhancement mean: " << this->enhancement_mean << std::endl;
    std::cout << "pauli potential mean: " << this->pauli_mean << std::endl;

    if (this->nvalidation > 0)
    {
        this->loadData(this->validation_dir, this->nx_vali, this->nvalidation, this->rho_vali, this->gamma_vali, this->p_vali, this->q_vali,
                        this->gammanl_vali, this->pnl_vali, this->qnl_vali, this->nablaRho_vali, this->enhancement_vali, this->enhancement_mean_vali, this->pauli_vali, this->pauli_mean_vali);
        this->input_vali = torch::zeros({this->nx_vali, this->ninput});
        if (this->nn_input_index["gamma"] >= 0) this->input_vali.index({"...", this->nn_input_index["gamma"]}) = gamma_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["p"] >= 0) this->input_vali.index({"...", this->nn_input_index["p"]}) = p_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["q"] >= 0) this->input_vali.index({"...", this->nn_input_index["q"]}) = q_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["gammanl"] >= 0) this->input_vali.index({"...", this->nn_input_index["gammanl"]}) = gammanl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["pnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["pnl"]}) = pnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["qnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["qnl"]}) = qnl_vali.reshape({this->nx_vali}).clone();
    }
    std::cout << "Load data done" << std::endl;
}

void Train::initData()
{
    this->nx = pow(this->fftdim, 3);
    this->nx_train = this->nx * this->ntrain;
    this->nx_vali = this->nx * this->nvalidation;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1}, {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}};

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
    this->pauli = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->pauli_mean = torch::zeros(this->ntrain);
    this->gamma = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->p = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->nablaRho = torch::zeros({this->ntrain, 3, this->fftdim, this->fftdim, this->fftdim});
    this->q = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->gammanl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->pnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    this->qnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    if (this->nvalidation > 0)
    {
        this->rho_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->enhancement_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->enhancement_mean_vali = torch::zeros(this->nvalidation);
        this->pauli_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->pauli_mean_vali = torch::zeros(this->nvalidation);
        this->gamma_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->p_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->nablaRho_vali = torch::zeros({this->nvalidation, 3, this->fftdim, this->fftdim, this->fftdim});
        this->q_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->gammanl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->pnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->qnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
    }

    if (this->ml_gamma || this->ml_gammanl){
        if (this->ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    }    
    if (this->ml_p || this->ml_pnl){
        if (this->ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    }
    if (this->ml_q || this->ml_qnl){
        if (this->ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    }
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
    torch::Tensor &enhancement,
    torch::Tensor &enhancement_mean,
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
        if (this->ml_gamma || this->ml_gammanl) this->loadTensor(dir[idata] + "/gamma.npy", cshape, fortran_order, container, idata, gamma);
        if (this->ml_gammanl) this->loadTensor(dir[idata] + "/gammanl.npy", cshape, fortran_order, container, idata, gammanl);
        if (this->ml_p || this->ml_pnl)
        {
            this->loadTensor(dir[idata] + "/p.npy", cshape, fortran_order, container, idata, p);
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhox.npy", cshape, fortran_order, container);
            nablaRho[idata][0] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoy.npy", cshape, fortran_order, container);
            nablaRho[idata][1] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoz.npy", cshape, fortran_order, container);
            nablaRho[idata][2] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
        }
        if (this->ml_pnl) this->loadTensor(dir[idata] + "/pnl.npy", cshape, fortran_order, container, idata, pnl);
        if (this->ml_q || this->ml_qnl) this->loadTensor(dir[idata] + "/q.npy", cshape, fortran_order, container, idata, q);
        if (this->ml_qnl) this->loadTensor(dir[idata] + "/qnl.npy", cshape, fortran_order, container, idata, qnl);

        this->loadTensor(dir[idata] + "/enhancement.npy", cshape, fortran_order, container, idata, enhancement);
        enhancement_mean[idata] = torch::mean(enhancement[idata]);

        if (this->loss == "potential" || this->loss == "both")
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
