#ifndef INPUT_H
#define INPUT_H

#include <torch/torch.h>

class Input
{
    // ---------- read in the settings from nnINPUT --------
  public:
    Input(){};
    ~Input()
    {
        delete[] this->train_dir;
        delete[] this->train_cell;
        delete[] this->train_a;
        delete[] this->validation_dir;
        delete[] this->validation_cell;
        delete[] this->validation_a;
    };

    void readInput();

    template <class T> static void read_value(std::ifstream &ifs, T &var)
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
    double kernel_scaling = 1.;
    double yukawa_alpha = 1.;

    // GPU
    std::string device_type = "gpu";
    bool check_pot = false;
};
#endif