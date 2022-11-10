#include "./train.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

void Train::loadData()
{
    this->loadData(this->train_dir, this->nx_train, this->rho, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl, this->nablaRho, this->enhancement);
    if (this->nvalidation > 0)
    {
        this->loadData(this->validation_dir, this->nx_vali, this->rho_vali, this->gamma_vali, this->p_vali, this->q_vali,
                        this->gammanl_vali, this->pnl_vali, this->qnl_vali, this->nablaRho_vali, this->enhancement_vali);
        this->input_vali = torch::zeros({this->nx_vali, this->ninput});
        if (this->nn_input_index["gamma"] >= 0) this->input_vali.index({"...", this->nn_input_index["gamma"]}) = gamma_vali.clone();
        if (this->nn_input_index["p"] >= 0) this->input_vali.index({"...", this->nn_input_index["p"]}) = p_vali.clone();
        if (this->nn_input_index["q"] >= 0) this->input_vali.index({"...", this->nn_input_index["q"]}) = q_vali.clone();
        if (this->nn_input_index["gammanl"] >= 0) this->input_vali.index({"...", this->nn_input_index["gammanl"]}) = gammanl_vali.clone();
        if (this->nn_input_index["pnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["pnl"]}) = pnl_vali.clone();
        if (this->nn_input_index["qnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["qnl"]}) = qnl_vali.clone();
    }
    std::cout << "Load train set done" << std::endl;
}

void Train::loadData(
    std::string dir, 
    int nx_train,
    torch::Tensor &rho,
    torch::Tensor &gamma,
    torch::Tensor &p,
    torch::Tensor &q,
    torch::Tensor &gammanl,
    torch::Tensor &pnl,
    torch::Tensor &qnl,
    torch::Tensor &nablaRho,
    torch::Tensor &enhancement
)
{
    std::vector<long unsigned int> cshape = {(long unsigned) nx_train};
    std::vector<double> container(nx_train);
    bool fortran_order = false;
    // npy::LoadArrayFromNumpy(dir+"/rho.npy", cshape, fortran_order, container);
    // rho = torch::tensor(container);
    this->loadTensor(dir + "/rho.npy", cshape, fortran_order, container, rho);
    if (this->ml_gamma || this->ml_gammanl) this->loadTensor(dir + "/gamma.npy", cshape, fortran_order, container, gamma);
    if (this->ml_gammanl) this->loadTensor(dir + "/gammanl.npy", cshape, fortran_order, container, gammanl);
    if (this->ml_p || this->ml_pnl)
    {
        this->loadTensor(dir + "/p.npy", cshape, fortran_order, container, p);
        npy::LoadArrayFromNumpy(dir + "/nablaRhox.npy", cshape, fortran_order, container);
        nablaRho[0] = torch::tensor(container);
        npy::LoadArrayFromNumpy(dir + "/nablaRhoy.npy", cshape, fortran_order, container);
        nablaRho[1] = torch::tensor(container);
        npy::LoadArrayFromNumpy(dir + "/nablaRhoz.npy", cshape, fortran_order, container);
        nablaRho[2] = torch::tensor(container);
    }
    if (this->ml_pnl) this->loadTensor(dir + "/pnl.npy", cshape, fortran_order, container, pnl);
    if (this->ml_q || this->ml_qnl) this->loadTensor(dir + "/q.npy", cshape, fortran_order, container, q);
    if (this->ml_qnl) this->loadTensor(dir + "/qnl.npy", cshape, fortran_order, container, qnl);

    this->loadTensor(dir + "/enhancement.npy", cshape, fortran_order, container, enhancement);
    enhancement.resize_({nx_train, 1});
}

void Train::loadTensor(
    std::string file,
    std::vector<long unsigned int> cshape,
    bool fortran_order, 
    std::vector<double> &container,
    torch::Tensor &data
)
{
    npy::LoadArrayFromNumpy(file, cshape, fortran_order, container);
    data = torch::tensor(container);
}

void Train::dumpTensor(const torch::Tensor &data, std::string filename, int nx)
{
    std::cout << "Dumping " << filename << std::endl;
    std::vector<double> v(nx);
    for (int ir = 0; ir < nx; ++ir) v[ir] = data[ir].item<double>();
    // std::vector<double> v(data.data_ptr<float>(), data.data_ptr<float>() + data.numel()); // this works, but only supports float tensor
    const long unsigned cshape[] = {(long unsigned) nx}; // shape
    npy::SaveArrayAsNumpy(filename, false, 1, cshape, v);
}
