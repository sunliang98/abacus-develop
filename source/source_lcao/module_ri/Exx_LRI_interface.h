#ifndef EXX_LRI_INTERFACE_H
#define EXX_LRI_INTERFACE_H

#include "Exx_LRI.h"
#include "source_lcao/module_ri/Mix_DMk_2D.h"
#include "source_lcao/module_ri/module_exx_symmetry/symmetry_rotation.h"
#include <memory>

class LCAO_Matrix;
class Charge_Mixing;
namespace elecstate
{
    class ElecState;
    template <typename TK, typename TR>
    class DensityMatrix;

    /// for symmetry, multi-k, nspin<4: restore DM(k) form DM(k_ibz)
    std::vector<std::vector<std::complex<double>>> restore_dm(const K_Vectors& kv,
        const std::vector<std::vector<std::complex<double>>>& dm_k_ibz,
        const ModuleSymmetry::Symmetry_rotation& symrot,
        const Parallel_2D& pv);
    /// do nothing if gamma_only
    std::vector<std::vector<double>> restore_dm(const K_Vectors& kv,
        const std::vector<std::vector<double>>& dm_k_ibz,
        const ModuleSymmetry::Symmetry_rotation& symrot,
        const Parallel_2D& pv);

}

template<typename T, typename Tdata>
class Exx_LRI_Interface
{
public:
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<TA, TC>;

    /// @brief  Constructor for Exx_LRI_Interface
    Exx_LRI_Interface(const Exx_Info::Exx_Info_RI& info)
    {
        this->exx_ptr = std::make_shared<Exx_LRI<Tdata>>(info);
    }
    Exx_LRI_Interface() = delete;

    ///// read and write Hexxs using cereal
    //void write_Hexxs_cereal(const std::string& file_name) const;
    //void read_Hexxs_cereal(const std::string& file_name);

    std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& get_Hexxs() const { return this->exx_ptr->Hexxs; }
    double &get_Eexx() const { return this->exx_ptr->Eexx; }
    ModuleBase::matrix &get_force() const { return this->exx_ptr->force_exx; }
    ModuleBase::matrix &get_stress() const { return this->exx_ptr->stress_exx; }

    // Processes in ESolver_KS_LCAO
    /// @brief in init: Exx_LRI::init()
    void init(const MPI_Comm &mpi_comm,
              const UnitCell &ucell,
              const K_Vectors &kv,
              const LCAO_Orbitals& orb);

    /// @brief: in cal_exx_ions: Exx_LRI::cal_exx_ions()
    void cal_exx_ions(const UnitCell& ucell, const bool write_cv = false);

    /// @brief: in cal_exx_elec: Exx_LRI::cal_exx_elec()
    void cal_exx_elec(const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& Ds,
                      const UnitCell& ucell,
                      const Parallel_Orbitals& pv,
                      const ModuleSymmetry::Symmetry_rotation* p_symrot = nullptr);

    /// @brief: in cal_exx_force: Exx_LRI::cal_exx_force()
    void cal_exx_force(const int& nat);

    /// @brief: in cal_exx_stress: Exx_LRI::cal_exx_stress()
    void cal_exx_stress(const double& omega, const double& lat0);

    // Processes in ESolver_KS_LCAO
    /// @brief in before_all_runners: set symmetry according to irreducible k-points
    /// since k-points are not reduced again after the variation of the cell and exx-symmetry must be consistent with k-points.
    /// In the future, we will reduce k-points again during cell-relax, then this setting can be moved to `exx_beforescf`.
    void exx_before_all_runners(const K_Vectors& kv, const UnitCell& ucell, const Parallel_2D& pv);

    /// @brief in beforescf: set xc type, opt_orb, do DM mixing
    void exx_beforescf(const int istep, const K_Vectors& kv, const Charge_Mixing& chgmix, const UnitCell& ucell, const LCAO_Orbitals& orb);

    /// @brief in eachiterinit:  do DM mixing and calculate Hexx when entering 2nd SCF
    void exx_eachiterinit(const int istep,
                          const UnitCell& ucell,
                          const elecstate::DensityMatrix<T, double>& dm/**< double should be Tdata if complex-PBE-DM is supported*/,
                          const K_Vectors& kv,
                          const int& iter);

    /// @brief in hamilt2rho: calculate Hexx and Eexx
    void exx_hamilt2rho(elecstate::ElecState& elec, const Parallel_Orbitals& pv, const int iter);

    /// @brief in iter_finish: write Hexx, do something according to whether SCF is converged
    void exx_iter_finish(const K_Vectors& kv,
                         const UnitCell& ucell,
                         hamilt::Hamilt<T>& hamilt,
                         elecstate::ElecState& elec,
                         Charge_Mixing& chgmix,
                         const double& scf_ene_thr,
                         int& iter,
                         const int istep,
                         bool& conv_esolver);
    /// @brief: in do_after_converge: add exx operators; do DM mixing if seperate loop
    bool exx_after_converge(const UnitCell& ucell,
                            hamilt::Hamilt<T>& hamilt,
                            const elecstate::DensityMatrix<T, double>& dm/**< double should be Tdata if complex-PBE-DM is supported*/,
                            const K_Vectors& kv,
                            const int& nspin,
                            int& iter,
                            const int& istep,
                            const double& etot,
                            const double& scf_ene_thr);

    int two_level_step = 0;
    double etot_last_outer_loop = 0.0;
    elecstate::DensityMatrix<T, double>* dm_last_step;

    std::shared_ptr<Exx_LRI<Tdata>> exx_ptr;

private:
    Mix_DMk_2D mix_DMk_2D;

    bool exx_spacegroup_symmetry = false;
    ModuleSymmetry::Symmetry_rotation symrot_;

    struct Flag_Finish
    {
        bool init = false;
        bool ions = false;
        bool elec = false;
        bool force = false;
        bool stress = false;
    };
    Flag_Finish flag_finish;
};

#include "Exx_LRI_interface.hpp"

#endif
