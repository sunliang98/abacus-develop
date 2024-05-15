#ifndef ML_DATA_H
#define ML_DATA_H

#include <vector>
#include "kedf_wt.h"
#include "kedf_tf.h"
#include "module_elecstate/elecstate_pw.h"

class ML_data{
public:
    ~ML_data()
    {
        for (int ik = 0; ik < this->nkernel; ++ik) delete[] this->kernel[ik];
        delete[] this->kernel;
        delete[] this->chi_xi;
        delete[] this->chi_pnl;
        delete[] this->chi_qnl;
        delete[] this->kernel_type;
        delete[] this->kernel_scaling;
        delete[] this->yukawa_alpha;
    }

    void set_para(
        const int &nx,
        const double &nelec, 
        const double &tf_weight, 
        const double &vw_weight,
        const double &chi_p,
        const double &chi_q,
        const std::string &chi_xi_,
        const std::string &chi_pnl_,
        const std::string &chi_qnl_,
        const int &nkernel,
        const std::string &kernel_type_,
        const std::string &kernel_scaling_,
        const std::string &yukawa_alpha_,
        const std::string &kernel_file_,
        const double &mu,
        const double &n_max,
        ModulePW::PW_Basis *pw_rho,
        const UnitCell& ucell);
    // output all parameters
    void generateTrainData_WT(
        const double * const *prho, 
        KEDF_WT &wt, 
        KEDF_TF &tf, 
        ModulePW::PW_Basis *pw_rho,
        const double *veff
    );
    void generateTrainData_KS(
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        const double *veff
    );
    void generateTrainData_KS(
        psi::Psi<std::complex<float>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        const double *veff
    ){} // a mock function
    void generate_descriptor(
        const double * const *prho, 
        ModulePW::PW_Basis *pw_rho,
        std::vector<std::vector<double>> &nablaRho
    );
    // get input parameters
    void getGamma(const double * const *prho, std::vector<double> &rgamma);
    void getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp);
    void getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq);
    void getGammanl(const int ikernel, std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl);
    void getPnl(const int ikernel, std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl);
    void getQnl(const int ikernel, std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl);
    // new parameters 2023-02-03
    void getXi(std::vector<double> &pgamma, std::vector<double> &pgammanl, std::vector<double> &rxi);
    void getTanhXi(const int ikernel, std::vector<double> &pgamma, std::vector<double> &pgammanl, std::vector<double> &rtanhxi);
    void getTanhP(std::vector<double> &pp, std::vector<double> &rtanhp);
    void getTanhQ(std::vector<double> &pq, std::vector<double> &rtanhq);
    void getTanh_Pnl(const int ikernel, std::vector<double> &ppnl, std::vector<double> &rtanh_pnl);
    void getTanh_Qnl(const int ikernel, std::vector<double> &pqnl, std::vector<double> &rtanh_qnl);
    void getTanhP_nl(const int ikernel, std::vector<double> &ptanhp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhp_nl);
    void getTanhQ_nl(const int ikernel, std::vector<double> &ptanhq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhq_nl);
    void getfP(std::vector<double> &pp, std::vector<double> &rfp);
    void getfQ(std::vector<double> &pq, std::vector<double> &rfq);
    void getfP_nl(const int ikernel, std::vector<double> &pfp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rfp_nl);
    void getfQ_nl(const int ikernel, std::vector<double> &pfq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rfq_nl);
    // 2023-03-20
    void getTanhXi_nl(const int ikernel, std::vector<double> &ptanhxi, ModulePW::PW_Basis *pw_rho, std::vector<double> &rtanhxi_nl);
    // get target
    void getF_WT(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho,std::vector<double> &rF);
    void getPauli_WT(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpauli);

    void getF_KS1(
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        const std::vector<std::vector<double>> &nablaRho,
        std::vector<double> &rF,
        std::vector<double> &rpauli
    );
    void getF_KS2(
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        std::vector<double> &rF,
        std::vector<double> &rpauli
    );
    // get intermediate variables of V_Pauli
    void getNablaRho(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &rnablaRho);

    void get_r_matrix(const UnitCell &ucell, ModulePW::PW_Basis *pw_rho, const double rcut, const int n_max, std::vector<double> &r_matrix);
    double soft(const double norm, const double r_cut);

    // tools
    double MLkernel(double eta, double tf_weight, double vw_weight);
    double MLkernel_yukawa(double eta, double alpha);
    void read_kernel(const std::string &fileName, const double& scaling, ModulePW::PW_Basis *pw_rho, double* kernel_);
    void multiKernel(const int ikernel, double *pinput, ModulePW::PW_Basis *pw_rho, double *routput);
    void Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    void divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    // void dumpTensor(const torch::Tensor &data, std::string filename);
    void loadVector(std::string filename, std::vector<double> &data);
    void dumpVector(std::string filename, const std::vector<double> &data);

    void tanh(std::vector<double> &pinput, std::vector<double> &routput, double chi=1.);
    double dtanh(double tanhx, double chi=1.);
    void f(std::vector<double> &pinput, std::vector<double> &routput);

    // new parameters 2023-02-13
    double* chi_xi = nullptr;
    double chi_p = 1.;
    double chi_q = 1.;
    double* chi_pnl = nullptr;
    double* chi_qnl = nullptr;

    int nx = 0;
    double dV = 0.;
    double rho0 = 0.;
    double kF = 0.;
    double tkF = 0.;
    double alpha = 5./6.;
    double beta = 5./6.;
    // double weightml = 1.;
    const double cTF = 3.0/10.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0) * 2; // 10/3*(3*pi^2)^{2/3}, multiply by 2 to convert unit from Hartree to Ry, finally in Ry*Bohr^(-2)
    const double pqcoef = 1.0 / (4.0 * pow(3*pow(M_PI, 2.0), 2.0/3.0)); // coefficient of p and q
    
    int nkernel = 1;
    int *kernel_type = nullptr;
    double *kernel_scaling = nullptr;
    double *yukawa_alpha = nullptr;
    std::string *kernel_file = nullptr;
    double **kernel = nullptr;

    // descriptor of position 2024-04-22
    double mu = 1.0;
    double rcut = 1.0;
    int n_max = 1;

    template<class T>
    void split_string(const std::string &input, const int &length, const T &default_, T* &output)
    {
        if (output == nullptr)
        {
            output = new T[length];
        }

        std::stringstream input_string;
        input_string << input;
        std::stringstream convert_string;
        std::string temp = "";
        int i = 0;
        while (std::getline(input_string, temp, '_') && i < length)
        {
            convert_string << temp;
            convert_string >> output[i];
            convert_string.clear();
            ++i;
        }
        for (int j = i; j < length; ++j)
        {
            output[j] = default_;
        }
    }

    std::string file_name(std::string parameter, const int kernel_type, const double kernel_scaling);
};

#endif