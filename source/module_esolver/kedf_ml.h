// #include <stdio.h>
// #include <math.h>
#include <vector>
#include "./kedf_wt.h"
#include "./kedf_tf.h"
// #include "npy.hpp"
// #include "/home/dell/2_software/libnpy/libnpy/include/npy.hpp"
#include "./ml-of/nn_of.h"
// #include "../module_base/timer.h"
// #include "../module_base/global_function.h"
// #include "../module_base/global_variable.h"
// #include "../module_base/matrix.h"
// #include "../module_pw/pw_basis.h"

class KEDF_ML
{
public:
    KEDF_ML()
    {
        this->kernel = NULL;
        // this->stress.create(3,3);
    }
    ~KEDF_ML()
    {
        if (this->kernel != NULL) delete[] this->kernel;
    }

    void set_para(int nx, double dV, double nelec, double tf_weight, double vw_weight, ModulePW::PW_Basis *pw_rho);

    double get_energy(const double * const * prho, ModulePW::PW_Basis *pw_rho);
    // double get_energy_density(const double * const *prho, int is, int ir, ModulePW::PW_Basis *pw_rho);
    void ML_potential(const double * const * prho, ModulePW::PW_Basis *pw_rho, ModuleBase::matrix &rpotential);
    // void get_stress(double cellVol, const double * const * prho, ModulePW::PW_Basis *pw_rho, double vw_weight);
    // double diffLinhard(double eta, double vw_weight);

    // output all parameters
    void generateTrainData(const double * const *prho, KEDF_WT &wt, KEDF_TF &tf, ModulePW::PW_Basis *pw_rho);
    void localTest(const double * const *prho, ModulePW::PW_Basis *pw_rho);
    // get input parameters
    void getGamma(const double * const *prho, std::vector<double> &rgamma);
    void getP(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &pnablaRho, std::vector<double> &rp);
    void getQ(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rq);
    void getGammanl(std::vector<double> &pgamma, ModulePW::PW_Basis *pw_rho, std::vector<double> &rgammanl);
    void getPnl(std::vector<double> &pp, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpnl);
    void getQnl(std::vector<double> &pq, ModulePW::PW_Basis *pw_rho, std::vector<double> &rqnl);
    // get target
    void getPauli(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rpauli);
    void getF(KEDF_WT &wt, KEDF_TF &tf, const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rF);
    // get intermediate variables of V_Pauli
    void getNablaRho(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<std::vector<double>> &rnablaRho);

    // potentials
    double potGammaTerm(int ir);
    double potPTerm1(int ir);
    double potQTerm1(int ir);
    void potGammanlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rGammanlTerm);
    void potPPnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rPPnlTerm);
    void potQQnlTerm(const double * const *prho, ModulePW::PW_Basis *pw_rho, std::vector<double> &rQQnlTerm);

    // tools
    double MLkernel(double eta, double tf_weight, double vw_weight);
    void multiKernel(double *pinput, ModulePW::PW_Basis *pw_rho, double *routput);
    void Laplacian(double * pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    void divergence(double ** pinput, ModulePW::PW_Basis *pw_rho, double * routput);
    void dumpTensor(const torch::Tensor &data, std::string filename);
    void updateInput(const double * const * prho, ModulePW::PW_Basis *pw_rho);

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
    double MLenergy = 0.;
    double *kernel;
    // ModuleBase::matrix stress;

    // informations about input
    int ninput = 0;
    std::map<std::string, int> nn_input_index;
    std::vector<double> gamma;
    std::vector<double> p;
    std::vector<double> q;
    std::vector<double> gammanl;
    std::vector<double> pnl;
    std::vector<double> qnl;
    std::vector<std::vector<double> > nablaRho;

    std::shared_ptr<NN_OFImpl> nn;
};