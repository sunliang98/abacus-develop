#ifndef TDEKINETIC_H
#define TDEKINETIC_H
#include "module_base/timer.h"
#include "operator_lcao.h"
#include "module_cell/klist.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_hamilt_lcao/hamilt_lcaodft/center2_orb-orb11.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "module_basis/module_ao/ORB_table_phi.h"
#include "module_basis/module_ao/ORB_gaunt_table.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"


namespace hamilt
{

#ifndef __TDEKINETICTEMPLATE
#define __TDEKINETICTEMPLATE

/// The TDEkinetic class template inherits from class T
/// It is used to calculate correction term of kinetic energy in time-dependent DFT
template <class T>
class TDEkinetic : public T
{
};

#endif

/// TDEkinetic class template specialization for OperatorLCAO<TK> base class
/// It is used to calculate correction term of kinetic energy in time-dependent DFT
/// Template parameters:
/// - TK: data type of k-space Hamiltonian
/// - TR: data type of real space Hamiltonian

template <typename TK, typename TR>
class TDEkinetic<OperatorLCAO<TK,TR>> : public OperatorLCAO<TK, TR>
{
  public:
    TDEkinetic<OperatorLCAO<TK, TR>>(LCAO_Matrix* LM_in,
                                 hamilt::HContainer<TR>* hR_in,
                                 std::vector<TK>* hK_in,
                                 hamilt::HContainer<TR>* SR_in,
                                 const K_Vectors* kv_in,
                                 const UnitCell* ucell_in,
                                 Grid_Driver* GridD_in);
    ~TDEkinetic();

    virtual void contributeHR() override;
    virtual void contributeHk(int ik) override;
    /// @brief init two center integrals and vector potential for td_ekintic term
    void init_td(void);

    /**
     * @brief initialize HR, search the nearest neighbor atoms
     * HContainer is used to store the non-local pseudopotential matrix with specific <I,J,R> atom-pairs
     * the size of HR will be fixed after initialization
     */
    void initialize_HR(Grid_Driver* GridD, const Parallel_Orbitals* paraV);

    /**
     * @brief initialize HR_tmp
     * Allocate the memory for HR_tmp with the same size as HR
    */
    void initialize_HR_tmp(const Parallel_Orbitals* paraV);

    /**
     * @brief calculate the HR local matrix of <I,J,R> atom pair
     */
    void cal_HR_IJR(const int& iat1,const int& iat2,const Parallel_Orbitals* paraV,const ModuleBase::Vector3<double>& dtau,std::complex<double>* data_pointer,TR* s_pointer);

    /**
     * @brief calculate the ekinetic matrix correction term in tddft with specific <I,J,R> atom-pairs
     * nearest neighbor atoms don't need to be calculated again
     * loop the atom-pairs in HR and calculate the ekinetic matrix
     */
    void calculate_HR(void);
    virtual void set_HR_fixed(void*)override;

    TD_Velocity td_velocity;

  private:
    const UnitCell* ucell = nullptr;
    
    HContainer<TR>* SR = nullptr;
    /// @brief Store real space hamiltonian. TD term should include imaginary part, thus it has to be complex type. Only shared between TD operators.
    HContainer<std::complex<double>>* hR_tmp = nullptr;
    Grid_Driver* Grid = nullptr;

    const K_Vectors* kv;
    /// @brief correction term iA⋅∇
    void td_ekinetic_scalar(std::complex<double>* Hloc, TR* Sloc, int nnr);
    /// @brief correction term A^2*S
    void td_ekinetic_grad(std::complex<double>* Hloc, int nnr, ModuleBase::Vector3<double> grad_overlap);

    ORB_table_phi MOT;
	  ORB_gaunt_table MGT;
    
    /// @brief Store the two center integrals outcome <𝝓_𝝁𝟎 |𝛁| 𝝓_𝝂𝑹> for td_ekinetic term
    std::map<size_t,                                // TA
      std::map<size_t,                            // TB
        std::map<int,                           // LA
          std::map<size_t,                    // NA
            std::map<int,                   // LB
              std::map<size_t,            // NB
                Center2_Orb::Orb11>>>>>> center2_orb11_s;

    /// @brief Store the vector potential for td_ekinetic term
    ModuleBase::Vector3<double> cart_At;

    /// @brief exact the nearest neighbor atoms from all adjacent atoms
    std::vector<AdjacentAtomInfo> adjs_all;

    bool hR_tmp_done = false;
    bool allocated = false;
    bool output_hR_done = false;
    bool out_mat_R = false;
};

} // namespace hamilt
#endif
