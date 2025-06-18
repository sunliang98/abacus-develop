#ifndef PEXSI_Solver_H
#define PEXSI_Solver_H

#include <vector>

namespace pexsi
{
class PEXSI_Solver
{
  public:
    void prepare(const int blacs_text,
                 const int nb,
                 const int nrow,
                 const int ncol,
                 const double* h,
                 const double* s,
                 double*& DM,
                 double*& EDM);
    int solve(double mu0);
    const double get_totalFreeEnergy() const;
    const double get_totalEnergyH() const;
    const double get_totalEnergyS() const;
    const double get_mu() const;

    //==========================================================
    // PEXSI related variables
    //==========================================================
    /** 
     * @brief  Number of terms in the pole expansion.
     */ 
    static int pexsi_npole;
    /** 
     * @brief  Whether inertia counting is used at the very beginning.
     */ 
    static bool pexsi_inertia;
    /** 
     * @brief  Maximum number of PEXSI iterations after each inertia counting procedure.
     */ 
    static int pexsi_nmax;
    /** 
     * @brief  Whether to construct PSelInv communication pattern.
     */ 
    static bool pexsi_comm;
    /** 
     * @brief  Whether to use symmetric storage space used by the Selected Inversion algorithm for symmetric matrices.  
     */ 
    static bool pexsi_storage;
    /** 
     * @brief  Ordering strategy for factorization and selected inversion. 
     */ 
    static int pexsi_ordering;
    /** 
     * @brief  row permutation strategy for factorization and selected inversion.  
     */ 
    static int pexsi_row_ordering;
    /** 
     * @brief  Number of processors for PARMETIS/PT-SCOTCH.  Only used if the ordering == 0.
     */ 
    static int pexsi_nproc;
    /** 
     * @brief  Matrix structure.
     * - = 0   : Unsymmetric matrix
     * - = 1   : Symmetric matrix (default).
     */ 
    static bool pexsi_symm;
    /** 
     * @brief  Transpose.
     * - = 0   : Factor non transposed matrix (default).
     * - = 1   : Factor transposed matrix.
     */ 
    static bool pexsi_trans;
    /** 
     * @brief  The pole expansion method to be used.
     * - = 1   : Cauchy Contour Integral method used.
     * - = 2   : Moussa optimized method.
     */ 
    static int pexsi_method;
    /** 
     * @brief  The point parallelizaion of PEXSI.
     * - = 2  : Recommend two points parallelization
     */ 
    static int pexsi_nproc_pole;
    /** 
     * @brief  Temperature, in the same unit as H 
     */ 
    static double pexsi_temp;
    /** 
     * @brief  Spectral gap. **Note** This can be set to be 0 in most cases.
     */ 
    static double pexsi_gap;
    /** 
     * @brief  An upper bound for the spectral radius of \f$S^{-1} H\f$.
     */ 
    static double pexsi_delta_e;
    /** 
     * @brief  Initial guess of lower bound for mu.
     */ 
    static double pexsi_mu_lower;
    /** 
     * @brief  Initial guess of upper bound for mu.
     */ 
    static double pexsi_mu_upper;
    /** 
     * @brief  Initial guess for mu (for the solver) (AG)
     */ 
    static double pexsi_mu;
    /** 
     * @brief  Stopping criterion in terms of the chemical potential for the inertia counting procedure.
     */ 
    static double pexsi_mu_thr;
    /** 
     * @brief  If the chemical potential is not in the initial interval, the interval is expanded by muInertiaExpansion.
     */ 
    static double pexsi_mu_expand;
    /** 
     * @brief  Safe guard criterion in terms of the chemical potential to reinvoke the inertia counting procedure.
     */ 
    static double pexsi_mu_guard;
    /** 
     * @brief  Stopping criterion of the %PEXSI iteration in terms of the number of electrons compared to numElectronExact.
     */ 
    static double pexsi_elec_thr;
    /** 
     * @brief  If the absolute value of CCS matrix element is less than this value, it will be considered as zero.
     */ 
    static double pexsi_zero_thr;

  private:
    int blacs_text;
    int nb;
    int nrow;
    int ncol;
    double* h;
    double* s;
    double* DM;
    double* EDM;
    double totalEnergyH;
    double totalEnergyS;
    double totalFreeEnergy;
    double mu;
};
} // namespace pexsi
#endif // PEXSI_Solver_H