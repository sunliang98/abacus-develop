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

    torch::Tensor lossFunction(torch::Tensor enhancement, torch::Tensor target, torch::Tensor coef = torch::ones(1));
    torch::Tensor lossFunction_new(torch::Tensor enhancement, torch::Tensor target, torch::Tensor weight, torch::Tensor coef = torch::ones(1));
    // double lostFunction(torch::Tensor potentialML, torch::Tensor target);
    // torch::Tensor potLossFunction()

    void train();
    void dump();

    // torch::Device device = torch::Device(torch::kCPU);
    torch::Device device = torch::Device(torch::kCUDA);

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
    // new parameters 2023-02-14
    torch::Tensor xi;
    torch::Tensor tanhxi;
    torch::Tensor tanhxi_nl; // 2023-03-20
    torch::Tensor tanhp;
    torch::Tensor tanhq;
    torch::Tensor tanh_pnl;
    torch::Tensor tanh_qnl;
    torch::Tensor tanhp_nl;
    torch::Tensor tanhq_nl;
    // target
    torch::Tensor enhancement;
    torch::Tensor pauli;
    torch::Tensor enhancement_mean;
    torch::Tensor tau_mean; // mean Pauli energy
    torch::Tensor pauli_mean;
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
    // new parameters 2023-02-14
    torch::Tensor xi_vali;
    torch::Tensor tanhxi_vali;
    torch::Tensor tanhxi_nl_vali; // 2023-03-20
    torch::Tensor tanhp_vali;
    torch::Tensor tanhq_vali;
    torch::Tensor tanh_pnl_vali;
    torch::Tensor tanh_qnl_vali;
    torch::Tensor tanhp_nl_vali;
    torch::Tensor tanhq_nl_vali;
    // target
    torch::Tensor enhancement_vali;
    torch::Tensor pauli_vali;
    torch::Tensor enhancement_mean_vali;
    torch::Tensor tau_mean_vali;
    torch::Tensor pauli_mean_vali;
    // fft grid
    std::vector<std::vector<torch::Tensor>> fft_grid_vali; // ntrain*3*fftdim*fftdim*fftdim
    std::vector<torch::Tensor> fft_gg_vali;
    std::vector<torch::Tensor> fft_kernel_vali;
    // others
    double *vali_volume = nullptr;
    // ------------------------------------

    // -------- free electron gas ---------
    torch::Tensor feg_inpt;
    torch::Tensor feg_predict;
    torch::Tensor feg_dFdgamma;
    // ------------------------------------

// ============= 1. train_input.cpp ===========
// ---------- read in the settings from nnINPUT --------
public:
    void readInput();

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
    std::string loss = "both";
    double exponent = 5.; // exponent of weight rho^{exponent/3.}
    int nepoch = 1000;
    // double step_length = 0.01;
    double lr_start = 0.01; // learning rate 2023-02-24
    double lr_end = 1e-4;
    int lr_fre = 5000;
    int dump_fre = 1;
    int print_fre = 1;
    bool ml_gamma = false;
    bool ml_p = false;
    bool ml_q = false;
    bool ml_gammanl = false;
    bool ml_pnl = false;
    bool ml_qnl = false;
    // new parameters 2023-02-14
    bool ml_xi = false;
    bool ml_tanhxi = false;
    bool ml_tanhxi_nl = false; // 2023-03-20
    bool ml_tanhp = false;
    bool ml_tanhq = false;
    bool ml_tanh_pnl = false;
    bool ml_tanh_qnl = false;
    bool ml_tanhp_nl = false;
    bool ml_tanhq_nl = false;
    double chi_xi = 1.;
    double chi_p = 1.;
    double chi_q = 1.;
    double chi_pnl = 1.;
    double chi_qnl = 1.;

    int feg_limit = 0; // Free Electron Gas
    int change_step = 0; // when feg_limit=3, change the output of net after change_step
    double feg3_correct = 0.541324854612918; // ln(e - 1)

    // coefficients in loss function
    double coef_e = 1.;
    double coef_p = 1.;
    double coef_feg_e = 1.;
    double coef_feg_p = 1.;

    // size of nn
    int nnode = 10;
    int nlayer = 3;

    // yukawa kernel
    int kernel_type = 1;
    double yukawa_alpha = 1.;
public:
    bool check_pot = false;

private:
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
    );
    
public:
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
public:
    void setUpFFT();
private:
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
    double MLkernel_yukawa(double eta, double alpha);

// ============= 4. train_pot.cpp ===============
public:
    void potTest();
    torch::Tensor getPot(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &tauTF,
        const torch::Tensor &gamma,
        const torch::Tensor &p,
        const torch::Tensor &q,
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &tanhxi_nl,
        const torch::Tensor &tanhp,
        const torch::Tensor &tanhq,
        const torch::Tensor &tanh_pnl,
        const torch::Tensor &tanh_qnl,
        const torch::Tensor &F,
        const torch::Tensor &gradient,
        const torch::Tensor &kernel,
        const std::vector<torch::Tensor> &grid,
        const torch::Tensor &gg
    );
private:

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
    torch::Tensor potGammanlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &gamma,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potPPnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potQQnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg    
    );


    torch::Tensor potXiTerm1(
        const torch::Tensor &xi,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxiTerm1(
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhpTerm1(
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhqTerm1(
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &gradient
    );
    torch::Tensor potXinlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxinlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &tanhxi,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhxi_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &xi,
        const torch::Tensor &tanhxi,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient
    );
    torch::Tensor potTanhpTanh_pnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &tanh_pnl,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potTanhqTanh_qnlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &tanh_qnl,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg
    );
    torch::Tensor potTanhpTanhp_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &nablaRho,
        const torch::Tensor &p,
        const torch::Tensor &tanhp,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor potTanhqTanhq_nlTerm(
        const torch::Tensor &rho,
        const torch::Tensor &q,
        const torch::Tensor &tanhq,
        const torch::Tensor &kernel,
        const torch::Tensor &tauTF,
        const torch::Tensor &gradient,
        const torch::Tensor &gg
    );

    // Tools for getting potential
    torch::Tensor divergence(
        const torch::Tensor &input,
        const std::vector<torch::Tensor> &grid
    );
    torch::Tensor Laplacian(
        const torch::Tensor &input,
        const torch::Tensor &gg
    );
    torch::Tensor dtanh(
        const torch::Tensor &tanhx,
        const double chi
    );

    const double cTF = 3.0/10.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0) * 2; // 10/3*(3*pi^2)^{2/3}, multiply by 2 to convert unit from Hartree to Ry, finally in Ry*Bohr^(-2)
    const double pqcoef = 1.0 / (4.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0)); // coefficient of p and q
};

// class OF_data : public torch::data::Dataset<OF_data>
// {
// private:
//     torch::Tensor input;
//     torch::Tensor target;

// public:
//     explicit OF_data(torch::Tensor &input, torch::Tensor &target)
//     {
//         this->input = input.clone();
//         this->target = target.clone();
//     }

//     torch::data::Example<> get(size_t index) override 
//     {
//         return {this->input[index], this->target[index]};
//     }

//     torch::optional<size_t> size() const override 
//     {
//         return this->input.size(0);
//     }
// };

#endif