#ifndef TRAIN
#define TRAIN

#include <torch/torch.h>
#include "./nn_of.h"

class Train{
public:
    Train(){};
    ~Train();
    
    std::shared_ptr<NN_OFImpl> nn;

    void init();

    torch::Tensor lossFunction(torch::Tensor enhancement, torch::Tensor target);
    // double lostFunction(torch::Tensor potentialML, torch::Tensor target);
    // torch::Tensor potLossFunction()

    void train();
    void dump();

    int nx = 1;
    int nx_train = 1;
    int nx_vali = 1;
    int ninput = 6;

    // torch::Device device = torch::Device(torch::kCUDA);

    //----------- training set -----------
    torch::Tensor rho;
    // inputs
    torch::Tensor gamma;
    torch::Tensor p;
    torch::Tensor q;
    torch::Tensor gammanl;
    torch::Tensor pnl;
    torch::Tensor qnl;
    torch::Tensor nablaRho;
    // target
    torch::Tensor enhancement;
    torch::Tensor pauli;
    // fft grid
    std::vector<std::vector<torch::Tensor>> fft_grid_train; // ntrain*3*fftdim*fftdim*fftdim
    std::vector<torch::Tensor> fft_gg_train;
    std::vector<torch::Tensor> fft_kernel_train;
    // others
    double *train_volume = nullptr;
    //------------------------------------

    //---------validation set ------------
    torch::Tensor rho_vali;
    // inputs
    torch::Tensor gamma_vali;
    torch::Tensor p_vali;
    torch::Tensor q_vali;
    torch::Tensor gammanl_vali;
    torch::Tensor pnl_vali;
    torch::Tensor qnl_vali;
    torch::Tensor nablaRho_vali;
    torch::Tensor input_vali;
    // target
    torch::Tensor enhancement_vali;
    torch::Tensor pauli_vali;
    // fft grid
    std::vector<std::vector<torch::Tensor>> fft_grid_vali; // ntrain*3*fftdim*fftdim*fftdim
    std::vector<torch::Tensor> fft_gg_vali;
    std::vector<torch::Tensor> fft_kernel_vali;
    // others
    double *vali_volume = nullptr;
    // ------------------------------------

// ============= 1. train_input.cpp ===========
// ---------- read in the settings from nnINPUT --------
public:
    void readInput();
private:
    template <class T>
    static void read_value(std::ifstream &ifs, T &var)
    {
        ifs >> var;
        ifs.ignore(150, '\n');
        return;
    }
    // input variables
    int fftdim = 0;
    int nbatch = 0;
    int ntrain = 1;
    int nvalidation = 0;
    std::string *train_dir = nullptr;
    std::string *train_cell = nullptr;
    double *train_a = nullptr;
    std::string *validation_dir = nullptr;
    std::string *validation_cell = nullptr;
    double *validation_a = nullptr;
    std::string loss = "energy";
    int nepoch = 1000;
    double step_length = 0.01;
    int dump_fre = 1;
    int print_fre = 1;
    bool ml_gamma = false;
    bool ml_p = false;
    bool ml_q = false;
    bool ml_gammanl = false;
    bool ml_pnl = false;
    bool ml_qnl = false;

    std::map<std::string, int> nn_input_index;

// =========== 2. train_data.cpp ===========
// --------- load the data from .npy files ------
public:
    void loadData();
private:
    void initData();
    void loadData(
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
    );
    void loadTensor(
        std::string file,
        std::vector<long unsigned int> cshape,
        bool fortran_order, 
        std::vector<double> &container,
        int index,
        torch::Tensor &data
    );
// -------- dump Tensor into .npy files ---------
    void dumpTensor(const torch::Tensor &data, std::string filename, int nx);

// ============== 3. train_ff.cpp ==============
// ------------ set up grid of FFT ----------
private:
    void setUpFFT();
    void initGrid();
    void initGrid_(
        const int fftdim,
        const int nstru,
        const std::string *cell,
        const double *a,
        double *volume,
        std::vector<std::vector<torch::Tensor>> &grid, 
        std::vector<torch::Tensor> &gg
    );
    void initScRecipGrid(
        const int fftdim, 
        const double a, 
        const int index, 
        double *volume,
        std::vector<std::vector<torch::Tensor>> &grid, 
        std::vector<torch::Tensor> &gg
    );
    void initFccRecipGrid(
        const int fftdim, 
        const double a, 
        const int index, 
        double *volume,
        std::vector<std::vector<torch::Tensor>> &grid, 
        std::vector<torch::Tensor> &gg
    );
    void initBccRecipGrid(
        const int fftdim, 
        const double a, 
        const int index, 
        double *volume,
        std::vector<std::vector<torch::Tensor>> &grid, 
        std::vector<torch::Tensor> &gg
    );
// ------------ fill the kernel in reciprocal space ----------
    void fillKernel();
    void fiilKernel_(
        const int fftdim,
        const int nstru,
        const torch::Tensor &rho,
        const double* volume,
        const std::string *cell,
        const std::vector<torch::Tensor> &fft_gg,
        std::vector<torch::Tensor> &fft_kernel
    );
    double MLkernel(double eta, double tf_weight = 1., double vw_weight = 1.);

// ============= 4. train_pot.cpp ===============
public:
    void potTest();
private:

    torch::Tensor getPot(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &tauTF,
        const torch::Tensor &gamma,
        const torch::Tensor &p,
        const torch::Tensor &q,
        const torch::Tensor &F,
        const torch::Tensor &gradient,
        const torch::Tensor &kernel,
        const std::vector<torch::Tensor> &grid,
        const torch::Tensor &gg
    );

    torch::Tensor potGammaTerm(
        const torch::Tensor &gamma,
        const torch::Tensor &gradient
    );
    torch::Tensor potPTerm1(
        const torch::Tensor &p,
        const torch::Tensor &gradient
    );
    torch::Tensor potQTerm1(
        const torch::Tensor &q,
        const torch::Tensor &gradient
    );
    // void potGammanlTerm(const torch::Tensor &rho, torch::Tensor &rGammanlTerm);
    torch::Tensor potGammanlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &gamma,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    // void potPPnlTerm(const torch::Tensor &rho, torch::Tensor &rPPnlTerm);
    torch::Tensor potPPnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    // void potQQnlTerm(const torch::Tensor &rho, torch::Tensor &rQQnlTerm);
    torch::Tensor potQQnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg    
    );

    // Tools for getting potential
    torch::Tensor multiKernel(
        const torch::Tensor &pinput,
        const torch::Tensor &kernel
    );
    // void Laplacian(torch::Tensor &pinput, torch::Tensor &routput);
    // void divergence(torch::Tensor &pinput, torch::Tensor  &routput);
    torch::Tensor divergence(
        const torch::Tensor &input,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor Laplacian(
        const torch::Tensor &input,
        const torch::Tensor &gg
    );

    const double cTF = 3.0/10.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0) * 2; // 10/3*(3*pi^2)^{2/3}, multiply by 2 to convert unit from Hartree to Ry, finally in Ry*Bohr^(-2)
    const double pqcoef = 1.0 / (4.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0)); // coefficient of p and q
};

class OF_data : public torch::data::Dataset<OF_data>
{
private:
    torch::Tensor input;
    torch::Tensor target;

public:
    explicit OF_data(torch::Tensor &input, torch::Tensor &target)
    {
        this->input = input.clone();
        this->target = target.clone();
    }

    torch::data::Example<> get(size_t index) override 
    {
        return {this->input[index], this->target[index]};
    }

    torch::optional<size_t> size() const override 
    {
        return this->input.size(0);
    }
};

#endif