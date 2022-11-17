#include "./train.h"
#include <math.h>

void Train::setUpFFT()
{
    this->initGrid();
    this->fillKernel();
    // this->dumpTensor(this->fft_kernel_train[0].reshape({this->nx}), "kernel_fcc.npy", this->nx);
    // this->dumpTensor(this->fft_kernel_vali[0].reshape({this->nx}), "kernel_bcc.npy", this->nx);
}

void Train::initGrid()
{
    this->initGrid_(this->fftdim, this->ntrain, this->train_cell, this->train_a, this->train_volume, this->fft_grid_train, this->fft_gg_train);
    if (this->nvalidation > 0)
    {
        this->initGrid_(this->fftdim, this->nvalidation, this->validation_cell, this->validation_a, this->vali_volume, this->fft_grid_vali, this->fft_gg_vali);
    }
}

void Train::initGrid_(
    const int fftdim,
    const int nstru,
    const std::string *cell,
    const double *a,
    double *volume,
    std::vector<std::vector<torch::Tensor>> &grid, 
    std::vector<torch::Tensor> &gg
)
{
    for (int it = 0; it < nstru; ++it)
    {
        if (cell[it] == "sc")
            this->initScRecipGrid(fftdim, a[it], it, volume, grid, gg);
        else if (cell[it] == "fcc") 
            this->initFccRecipGrid(fftdim, a[it], it, volume, grid, gg);
        else if (cell[it] == "bcc")
            this->initBccRecipGrid(fftdim, a[it], it, volume, grid, gg);
    }
}

void Train::initScRecipGrid(
    const int fftdim, 
    const double a, 
    const int index, 
    double *volume,
    std::vector<std::vector<torch::Tensor>> &grid, 
    std::vector<torch::Tensor> &gg
)
{
    volume[index] = pow(a, 3);
    torch::Tensor fre = torch::fft::fftfreq(fftdim, a/fftdim) * 2. * M_PI;
    grid[index] = torch::meshgrid({fre, fre, fre});
    gg[index] = grid[index][0] * grid[index][0] + grid[index][1] * grid[index][1] + grid[index][2] * grid[index][2];
}

void Train::initFccRecipGrid(
    const int fftdim, 
    const double a, 
    const int index, 
    double *volume,
    std::vector<std::vector<torch::Tensor>> &grid, 
    std::vector<torch::Tensor> &gg
)
{
    // std::cout << "init grid" << std::endl;
    volume[index] = pow(a, 3) / 4.;
    double coef = 1./sqrt(2.);
    // std::cout << "fftfreq" << std::endl;
    torch::Tensor fre = torch::fft::fftfreq(fftdim, a * coef/fftdim) * 2. * M_PI;
    auto originalGrid = torch::meshgrid({fre, fre, fre});
    grid[index][0] = coef * (-originalGrid[0] + originalGrid[1] + originalGrid[2]);
    grid[index][1] = coef * (originalGrid[0] - originalGrid[1] + originalGrid[2]);
    grid[index][2] = coef * (originalGrid[0] + originalGrid[1] - originalGrid[2]);
    // std::cout << "gg" << std::endl;
    gg[index] = grid[index][0] * grid[index][0] + grid[index][1] * grid[index][1] + grid[index][2] * grid[index][2];
    std::cout << "Init grid done" << std::endl;
}

void Train::initBccRecipGrid(
    const int fftdim, 
    const double a, 
    const int index, 
    double *volume,
    std::vector<std::vector<torch::Tensor>> &grid, 
    std::vector<torch::Tensor> &gg
)
{
    volume[index] = pow(a, 3) / 2.;
    double coef = sqrt(3.)/2.;
    torch::Tensor fre = torch::fft::fftfreq(fftdim, a * coef/fftdim) * 2. * M_PI;
    auto originalGrid = torch::meshgrid({fre, fre, fre});
    grid[index][0] = coef * (originalGrid[1] + originalGrid[2]);
    grid[index][1] = coef * (originalGrid[0] + originalGrid[2]);
    grid[index][2] = coef * (originalGrid[0] + originalGrid[1]);
    gg[index] = grid[index][0] * grid[index][0] + grid[index][1] * grid[index][1] + grid[index][2] * grid[index][2];
}

void Train::fillKernel()
{
    this->fiilKernel_(this->fftdim, this->ntrain, this->rho, this->train_volume, this->train_cell, this->fft_gg_train, this->fft_kernel_train);
    if (this->nvalidation > 0)
    {
        this->fiilKernel_(this->fftdim, this->nvalidation, this->rho_vali, this->vali_volume, this->validation_cell, this->fft_gg_vali, this->fft_kernel_vali);
    }
    std::cout << "Fill kernel done" << std::endl;
}

void Train::fiilKernel_(
    const int fftdim,
    const int nstru,
    const torch::Tensor &rho,
    const double* volume,
    const std::string *cell,
    const std::vector<torch::Tensor> &fft_gg,
    std::vector<torch::Tensor> &fft_kernel
)
{
    double rho0 = 0.;
    double tkF = 0.;
    double eta = 0.;
    for (int it = 0; it < nstru; ++it)
    {
        rho0 = torch::sum(rho[it]).item<double>()/this->nx;
        std::cout << "There are " << rho0 * volume[it] << " electrons in " << cell[it] << " strcture." << std::endl;
        tkF = 2. * pow(3. * pow(M_PI, 2) * rho0, 1./3.);
        fft_kernel[it] = torch::zeros({this->fftdim, this->fftdim, this->fftdim});
        for (int ix = 0; ix < this->fftdim; ++ix)
        {
            for (int iy = 0; iy < this->fftdim; ++iy)
            {
                for (int iz = 0; iz < this->fftdim; ++iz)
                {
                    eta = sqrt(fft_gg[it][ix][iy][iz].item<double>()) / tkF;
                    fft_kernel[it][ix][iy][iz] = this->MLkernel(eta);
                }
            }
        }
    }
}

double Train::MLkernel(double eta, double tf_weight, double vw_weight)
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
