#ifndef ML_DATA_H
#define ML_DATA_H
#include <vector>
#include "../module_hamilt/of_pw/kedf_wt.h"
#include "../module_hamilt/of_pw/kedf_tf.h"
#include "../module_elecstate/elecstate_pw.h"


class ML_data{
public:
    void set_para(int nx, double nelec, double tf_weight, double vw_weight, ModulePW::PW_Basis *pw_rho);
    // output all parameters
    void generateTrainData_WT(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf, ModulePW::PW_Basis *pw_rho);
    void generateTrainData_KS(
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        double *veff
    );
    // get input parameters
    void getGamma(const double * const *prho, std::vector<double> &rgamma);
    void getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp);
    void getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq);
    void getGammanl(std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl);
    void getPnl(std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl);
    void getQnl(std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl);
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

    // tools
    double MLkernel(double eta, double tf_weight, double vw_weight);
    void multiKernel(double *pinput, ModulePW::PW_Basis *pw_rho, double *routput);
    void Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    void divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    // void dumpTensor(const torch::Tensor &data, std::string filename);
    void loadVector(std::string filename, std::vector<double> &data);
    void dumpVector(std::string filename, const std::vector<double> &data);

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
    double *kernel = nullptr;

};

#endif