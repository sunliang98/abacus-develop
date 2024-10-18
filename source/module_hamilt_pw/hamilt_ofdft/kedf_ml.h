#ifndef KEDF_ML_H
#define KEDF_ML_H

#include "ml_data.h"

#include <vector>
#include "module_hamilt_pw/hamilt_ofdft/kedf_wt.h"
#include "module_hamilt_pw/hamilt_ofdft/kedf_tf.h"
#include "./ml_tools/nn_of.h"

class KEDF_ML
{
public:
    KEDF_ML()
    {
        // this->stress.create(3,3);
    }
    ~KEDF_ML()
    {
        delete this->ml_data;
        delete[] this->chi_xi;
        delete[] this->chi_pnl;
        delete[] this->chi_qnl;
    }

    void set_para(
        const int nx, 
        const double dV, 
        const double nelec, 
        const double tf_weight, 
        const double vw_weight, 
        const double chi_p,
        const double chi_q,
        const std::string chi_xi_,
        const std::string chi_pnl_,
        const std::string chi_qnl_,
        const int nnode,
        const int nlayer,
        const int &nkernel,
        const std::string &kernel_type_,
        const std::string &kernel_scaling_,
        const std::string &yukawa_alpha_,
        const std::string &kernel_file_,
        const bool &of_ml_gamma,
        const bool &of_ml_p,
        const bool &of_ml_q,
        const bool &of_ml_tanhp,
        const bool &of_ml_tanhq,
        const std::string &of_ml_gammanl_,
        const std::string &of_ml_pnl_,
        const std::string &of_ml_qnl_,
        const std::string &of_ml_xi_,
        const std::string &of_ml_tanhxi_,
        const std::string &of_ml_tanhxi_nl_,
        const std::string &of_ml_tanh_pnl_,
        const std::string &of_ml_tanh_qnl_,
        const std::string &of_ml_tanhp_nl_,
        const std::string &of_ml_tanhq_nl_,
        const std::string device_inpt,
        ModulePW::PW_Basis *pw_rho);

    void set_device(std::string device_inpt);

    double get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho);
    // double get_energy_density(const double * const *prho, int is, int ir, ModulePW::PW_Basis *pw_rho);
    void ml_potential(const double * const * prho, ModulePW::PW_Basis *pw_rho, ModuleBase::matrix &rpotential);
    // void get_stress(double cellVol, const double * const * prho, ModulePW::PW_Basis *pw_rho, double vw_weight);
    // double diffLinhard(double eta, double vw_weight);

    // output all parameters
    void generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf, ModulePW::PW_Basis *pw_rho, const double *veff);
    void localTest(const double * const *prho, ModulePW::PW_Basis *pw_rho);
    // get input parameters
    // void getGamma(const double * const *prho, std::vector<double> &rgamma);
    // void getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp);
    // void getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq);
    // void getGammanl(std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl);
    // void getPnl(std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl);
    // void getQnl(std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl);
    // // get target
    // void getPauli(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpauli);
    // void getF(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rF);
    // // get intermediate variables of V_Pauli
    // void getNablaRho(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &rnablaRho);

    // interface to NN
    void NN_forward(const double * const * prho, ModulePW::PW_Basis *pw_rho, bool cal_grad);

    void get_potential_(const double * const * prho, ModulePW::PW_Basis *pw_rho, ModuleBase::matrix &rpotential);

    // potentials
    double potGammaTerm(int ir);
    double potPTerm1(int ir);
    double potQTerm1(int ir);
    double potXiTerm1(int ir);
    double potTanhxiTerm1(int ir);
    double potTanhpTerm1(int ir);
    double potTanhqTerm1(int ir);
    void potGammanlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rGammanlTerm);
    void potXinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rXinlTerm);
    void potTanhxinlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxinlTerm);
    void potTanhxi_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhxi_nlTerm); // 2023-03-20 for tanhxi_nl
    void potPPnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rPPnlTerm);
    void potQQnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rQQnlTerm);
    void potTanhpTanh_pnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanh_pnlTerm);
    void potTanhqTanh_qnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanh_qnlTerm);
    void potTanhpTanhp_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhpTanhp_nlTerm);
    void potTanhqTanhq_nlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rTanhqTanhq_nlTerm);
    // tools
    // double MLkernel(double eta, double tf_weight, double vw_weight);
    // void multiKernel(double *pinput, ModulePW::PW_Basis *pw_rho, double *routput);
    // void Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    // void divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    void dumpTensor(const torch::Tensor &data, std::string filename);
    void dumpMatrix(const ModuleBase::matrix &data, std::string filename);
    void updateInput(const double * const * prho, ModulePW::PW_Basis *pw_rho);

    ML_data *ml_data = nullptr;

    int nx = 0;
    int nx_tot = 0.; // used to initialize nn_of
    double dV = 0.;
    double rho0 = 0.;
    double kF = 0.;
    double tkF = 0.;
    double alpha = 5./6.;
    double beta = 5./6.;
    // double weightml = 1.;
    const double cTF = 3.0/10.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0) * 2; // 10/3*(3*pi^2)^{2/3}, multiply by 2 to convert unit from Hartree to Ry, finally in Ry*Bohr^(-2)
    const double pqcoef = 1.0 / (4.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0)); // coefficient of p and q
    double ml_energy = 0.;
    // ModuleBase::matrix stress;
    double feg_net_F = 0.;
    double feg3_correct = 0.541324854612918; // ln(e - 1)

    // informations about input
    int ninput = 0;
    std::vector<double> gamma = {};
    std::vector<double> p = {};
    std::vector<double> q = {};
    std::vector<std::vector<double>> gammanl = {};
    std::vector<std::vector<double>> pnl = {};
    std::vector<std::vector<double>> qnl = {};
    std::vector<std::vector<double>> nablaRho = {};
    // new parameters 2023-02-13
    double* chi_xi = nullptr;
    double chi_p = 1.;
    double chi_q = 1.;
    std::vector<std::vector<double>> xi = {}; // we assume ONLY ONE of them is used.
    std::vector<std::vector<double>> tanhxi = {};
    std::vector<std::vector<double>> tanhxi_nl= {}; // 2023-03-20
    std::vector<double> tanhp = {};
    std::vector<double> tanhq = {};
    // plan 1
    double* chi_pnl = nullptr;
    double* chi_qnl = nullptr;
    std::vector<std::vector<double>> tanh_pnl = {};
    std::vector<std::vector<double>> tanh_qnl = {};
    // plan 2
    std::vector<std::vector<double>> tanhp_nl = {};
    std::vector<std::vector<double>> tanhq_nl = {};
    // GPU
    torch::DeviceType device_type = torch::kCPU;
    torch::Device device = torch::Device(torch::kCPU);
    torch::Device device_CPU = torch::Device(torch::kCPU);

    std::shared_ptr<NN_OFImpl> nn;
    double* enhancement_cpu_ptr = nullptr;
    double* gradient_cpu_ptr = nullptr;

    int nkernel = 1;

    // maps
    void init_data(
        const int &nkernel,
        const bool &of_ml_gamma,
        const bool &of_ml_p,
        const bool &of_ml_q,
        const bool &of_ml_tanhp,
        const bool &of_ml_tanhq,
        const std::string &of_ml_gammanl_,
        const std::string &of_ml_pnl_,
        const std::string &of_ml_qnl_,
        const std::string &of_ml_xi_,
        const std::string &of_ml_tanhxi_,
        const std::string &of_ml_tanhxi_nl_,
        const std::string &of_ml_tanh_pnl_,
        const std::string &of_ml_tanh_qnl_,
        const std::string &of_ml_tanhp_nl_,
        const std::string &of_ml_tanhq_nl_
    );
    
    bool ml_gamma = false;
    bool ml_p = false;
    bool ml_q = false;
    bool ml_tanhp = false;
    bool ml_tanhq = false;
    bool ml_gammanl = false;
    bool ml_pnl = false;
    bool ml_qnl = false;
    bool ml_xi = false;
    bool ml_tanhxi = false;
    bool ml_tanhxi_nl = false;
    bool ml_tanh_pnl = false;
    bool ml_tanh_qnl = false;
    bool ml_tanhp_nl = false;
    bool ml_tanhq_nl = false;

    std::vector<std::string> descriptor_type = {};
    std::vector<int> kernel_index = {};    
    std::map<std::string, std::vector<int>> descriptor2kernel = {};
    std::map<std::string, std::vector<int>> descriptor2index = {};
    std::map<std::string, std::vector<bool>> gene_data_label = {};

    torch::Tensor get_data(std::string parameter, const int ikernel);
};
#endif