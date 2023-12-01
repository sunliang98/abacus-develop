#include "./data.h"
#include "/home/xianyuer/data/1_sunliang/2_software/libnpy-old/include/npy.hpp"

void Data::loadData(Input &input, const int ndata, std::string *dir, const torch::Device device)
{
    this->initData(ndata, input.fftdim, device);
    this->loadData_(input, ndata, input.fftdim, dir);
    std::cout << "enhancement mean: " << this->enhancement_mean << std::endl;
    std::cout << "exponent: " << input.exponent << std::endl;
    std::cout << "tau mean: " << this->tau_mean << std::endl;
    std::cout << "pauli potential mean: " << this->pauli_mean << std::endl;
    std::cout << "Load data done" << std::endl;
}

void Data::initData(const int ndata, const int fftdim, const torch::Device device)
{
    this->nx = pow(fftdim, 3);
    this->nx_tot = this->nx * ndata;

    this->rho = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->enhancement = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->enhancement_mean = torch::zeros(ndata).to(device);
    this->tau_mean = torch::zeros(ndata).to(device);
    this->pauli = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->pauli_mean = torch::zeros(ndata).to(device);
    this->gamma = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->p = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->nablaRho = torch::zeros({ndata, 3, fftdim, fftdim, fftdim}).to(device);
    this->q = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->gammanl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->pnl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->qnl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->xi = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhxi = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhxi_nl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhp = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhq = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanh_pnl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanh_qnl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhp_nl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
    this->tanhq_nl = torch::zeros({ndata, fftdim, fftdim, fftdim}).to(device);
}

void Data::loadData_(
    Input &input,
    const int ndata,
    const int fftdim,
    std::string *dir 
)
{
    if (ndata <= 0) return;
    
    std::vector<long unsigned int> cshape = {(long unsigned) nx};
    std::vector<double> container(nx);
    bool fortran_order = false;

    for (int idata = 0; idata < ndata; ++idata)
    {
        this->loadTensor(dir[idata] + "/rho.npy", cshape, fortran_order, container, idata, fftdim, rho);
        if (input.ml_gamma || input.ml_gammanl || input.ml_xi || input.ml_tanhxi || input.ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/gamma.npy", cshape, fortran_order, container, idata, fftdim, gamma);
        }
        if (input.ml_gammanl || input.ml_xi || input.ml_tanhxi || input.ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/gammanl.npy", cshape, fortran_order, container, idata, fftdim, gammanl);
        }
        if (input.ml_p || input.ml_pnl || input.ml_tanhp || input.ml_tanh_pnl || input.ml_tanhp_nl)
        {
            this->loadTensor(dir[idata] + "/p.npy", cshape, fortran_order, container, idata, fftdim, p);
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhox.npy", cshape, fortran_order, container);
            nablaRho[idata][0] = torch::tensor(container).reshape({fftdim, fftdim, fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoy.npy", cshape, fortran_order, container);
            nablaRho[idata][1] = torch::tensor(container).reshape({fftdim, fftdim, fftdim});
            npy::LoadArrayFromNumpy(dir[idata] + "/nablaRhoz.npy", cshape, fortran_order, container);
            nablaRho[idata][2] = torch::tensor(container).reshape({fftdim, fftdim, fftdim});
        }
        if (input.ml_pnl || input.ml_tanh_pnl){
            this->loadTensor(dir[idata] + "/pnl.npy", cshape, fortran_order, container, idata, fftdim, pnl);
        }
        if (input.ml_q || input.ml_qnl || input.ml_tanhq || input.ml_tanh_qnl || input.ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/q.npy", cshape, fortran_order, container, idata, fftdim, q);
        }
        if (input.ml_qnl || input.ml_tanh_qnl){
            this->loadTensor(dir[idata] + "/qnl.npy", cshape, fortran_order, container, idata, fftdim, qnl);
        }
        if (input.ml_xi || input.ml_tanhxi || input.ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/xi.npy", cshape, fortran_order, container, idata, fftdim, xi);
        }
        if (input.ml_tanhxi || input.ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/tanhxi.npy", cshape, fortran_order, container, idata, fftdim, tanhxi);
        }
        if (input.ml_tanhxi_nl){
            this->loadTensor(dir[idata] + "/tanhxi_nl.npy", cshape, fortran_order, container, idata, fftdim, tanhxi_nl);
        }
        if (input.ml_tanhp || input.ml_tanhp_nl){
            this->loadTensor(dir[idata] + "/tanhp.npy", cshape, fortran_order, container, idata, fftdim, tanhp);
        }
        if (input.ml_tanhq || input.ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/tanhq.npy", cshape, fortran_order, container, idata, fftdim, tanhq);
        }
        if (input.ml_tanh_pnl){
            this->loadTensor(dir[idata] + "/tanh_pnl.npy", cshape, fortran_order, container, idata, fftdim, tanh_pnl);
        }
        if (input.ml_tanh_qnl){
            this->loadTensor(dir[idata] + "/tanh_qnl.npy", cshape, fortran_order, container, idata, fftdim, tanh_qnl);
        }
        if (input.ml_tanhp_nl){
            this->loadTensor(dir[idata] + "/tanhp_nl.npy", cshape, fortran_order, container, idata, fftdim, tanhp_nl);
        }
        if (input.ml_tanhq_nl){
            this->loadTensor(dir[idata] + "/tanhq_nl.npy", cshape, fortran_order, container, idata, fftdim, tanhq_nl);
        }

        this->loadTensor(dir[idata] + "/enhancement.npy", cshape, fortran_order, container, idata, fftdim, enhancement);
        enhancement_mean[idata] = torch::mean(enhancement[idata]);
        tau_mean[idata] = torch::mean(torch::pow(rho[idata], input.exponent/3.) * enhancement[idata]);

        if (input.loss == "potential" || input.loss == "both" || input.loss == "both_new")
        {
            this->loadTensor(dir[idata] + "/pauli.npy", cshape, fortran_order, container, idata, fftdim, pauli);
            pauli_mean[idata] = torch::mean(pauli[idata]);
        }
    }
    enhancement.resize_({this->nx_tot, 1});
    pauli.resize_({nx_tot, 1});
}

void Data::loadTensor(
    std::string file,
    std::vector<long unsigned int> cshape,
    bool fortran_order, 
    std::vector<double> &container,
    const int index,
    const int fftdim,
    torch::Tensor &data
)
{
    npy::LoadArrayFromNumpy(file, cshape, fortran_order, container);
    data[index] = torch::tensor(container).reshape({fftdim, fftdim, fftdim});
}

void Data::dumpTensor(const torch::Tensor &data, std::string filename, int nx)
{
    std::vector<double> v(nx);
    for (int ir = 0; ir < nx; ++ir) v[ir] = data[ir].item<double>();
    // std::vector<double> v(data.data_ptr<float>(), data.data_ptr<float>() + data.numel()); // this works, but only supports float tensor
    const long unsigned cshape[] = {(long unsigned) nx}; // shape
    npy::SaveArrayAsNumpy(filename, false, 1, cshape, v);
    std::cout << "Dumping " << filename << " done" << std::endl;
}
