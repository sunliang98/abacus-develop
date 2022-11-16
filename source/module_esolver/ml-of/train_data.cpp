#include "./train.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

void Train::loadData()
{
    this->loadData(this->train_dir, this->nx_train, this->ntrain, this->rho, this->gamma, this->p, this->q,
                   this->gammanl, this->pnl, this->qnl, this->nablaRho, this->enhancement, this->pauli);
    if (this->nvalidation > 0)
    {
        this->loadData(this->validation_dir, this->nx_vali, this->nvalidation, this->rho_vali, this->gamma_vali, this->p_vali, this->q_vali,
                        this->gammanl_vali, this->pnl_vali, this->qnl_vali, this->nablaRho_vali, this->enhancement_vali, this->pauli_vali);
        this->input_vali = torch::zeros({this->nx_vali, this->ninput});
        if (this->nn_input_index["gamma"] >= 0) this->input_vali.index({"...", this->nn_input_index["gamma"]}) = gamma_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["p"] >= 0) this->input_vali.index({"...", this->nn_input_index["p"]}) = p_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["q"] >= 0) this->input_vali.index({"...", this->nn_input_index["q"]}) = q_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["gammanl"] >= 0) this->input_vali.index({"...", this->nn_input_index["gammanl"]}) = gammanl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["pnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["pnl"]}) = pnl_vali.reshape({this->nx_vali}).clone();
        if (this->nn_input_index["qnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["qnl"]}) = qnl_vali.reshape({this->nx_vali}).clone();
    }
    std::cout << "Load train set done" << std::endl;
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
    torch::Tensor &pauli
)
{
    if (nDataSet <= 0) return;
    
    std::vector<long unsigned int> cshape = {(long unsigned) nx};
    std::vector<double> container(nx);
    bool fortran_order = false;

    for (int idata = 0; idata < nDataSet; ++idata)
    {
        this->loadTensor(dir[idata] + "/rho.npy", cshape, fortran_order, container, idata, rho);
        std::cout << "rho done" << std::endl;
        if (this->ml_gamma || this->ml_gammanl) this->loadTensor(dir[idata] + "/gamma.npy", cshape, fortran_order, container, idata, gamma);
        std::cout << "gamma done" << std::endl;
        if (this->ml_gammanl) this->loadTensor(dir[idata] + "/gammanl.npy", cshape, fortran_order, container, idata, gammanl);
        std::cout << "gammanl done" << std::endl;
        if (this->ml_p || this->ml_pnl)
        {
            this->loadTensor(dir[idata] + "/p.npy", cshape, fortran_order, container, idata, p);
            std::cout << "p done" << std::endl;
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhox.npy", cshape, fortran_order, container);
            nablaRho[idata][0] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoy.npy", cshape, fortran_order, container);
            nablaRho[idata][1] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoz.npy", cshape, fortran_order, container);
            nablaRho[idata][2] = torch::tensor(container).reshape({this->fftdim, this->fftdim, this->fftdim});
            std::cout << "nabla rho done" << std::endl;
        }
        if (this->ml_pnl) this->loadTensor(dir[idata] + "/pnl.npy", cshape, fortran_order, container, idata, pnl);
        std::cout << "pnl done" << std::endl;
        if (this->ml_q || this->ml_qnl) this->loadTensor(dir[idata] + "/q.npy", cshape, fortran_order, container, idata, q);
        std::cout << "q done" << std::endl;
        if (this->ml_qnl) this->loadTensor(dir[idata] + "/qnl.npy", cshape, fortran_order, container, idata, qnl);
        std::cout << "qnl done" << std::endl;

        this->loadTensor(dir[idata] + "/enhancement.npy", cshape, fortran_order, container, idata, enhancement);
        // enhancement.resize_({nx, 1});

        if (this->loss == "potential")
        {
            this->loadTensor(dir[idata] + "/pauli.npy", cshape, fortran_order, container, idata, pauli);
            // pauli.resize_({nx, 1});
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
