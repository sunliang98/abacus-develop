#include <torch/torch.h>
#include "./nn_of.h"
#include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"

class Train{
public:
    Train(){};
    ~Train();
    
    std::shared_ptr<NN_OFImpl> nn;

    // void setPara(int nx, int ninpt, int nbatch);

    void readInput();

    void loadData();

    void initNN();

    torch::Tensor lossFunction(torch::Tensor enhancement, torch::Tensor target);
    // double lostFunction(torch::Tensor potentialML, torch::Tensor target);

    void train();

    void dump();

    int nx = 1;
    int nx_vali = 1;
    int ninput = 6;

    // training set =============================
    std::vector<double> rho;
    // inputs
    std::vector<double> gamma;
    std::vector<double> p;
    std::vector<double> q;
    std::vector<double> gammanl;
    std::vector<double> pnl;
    std::vector<double> qnl;
    std::vector<std::vector<double> > nablaRho;
    // target
    torch::Tensor enhancement;
    // ==========================================

    // validation set ===========================
    std::vector<double> rho_vali;
    // inputs
    std::vector<double> gamma_vali;
    std::vector<double> p_vali;
    std::vector<double> q_vali;
    std::vector<double> gammanl_vali;
    std::vector<double> pnl_vali;
    std::vector<double> qnl_vali;
    std::vector<std::vector<double> > nablaRho_vali;
    // target
    torch::Tensor enhancement_vali;
    torch::Tensor input_vali;
    // ==========================================

private:
    template <class T>
    static void read_value(std::ifstream &ifs, T &var)
    {
        ifs >> var;
        ifs.ignore(150, '\n');
        return;
    }

    void loadData(
    std::string dir, 
    int nx,
    std::vector<double> &rho,
    std::vector<double> &gamma,
    std::vector<double> &p,
    std::vector<double> &q,
    std::vector<double> &gammanl,
    std::vector<double> &pnl,
    std::vector<double> &qnl,
    std::vector<std::vector<double> > &nablaRho,
    torch::Tensor &enhancement
    );

    // input variables
    int fftdim = 0;
    int nbatch = 0;
    int ntrain = 1;
    int nvalidation = 0;
    std::string train_dir = ".";
    std::string *train_cell = nullptr;
    double *train_a = nullptr;
    std::string validation_dir = ".";
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