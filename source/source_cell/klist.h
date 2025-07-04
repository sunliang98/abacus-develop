#ifndef K_VECTORS_H
#define K_VECTORS_H

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/matrix3.h"
#include "source_cell/unitcell.h"
#include "parallel_kpoints.h"
#include "k_vector_utils.h"
#include <vector>

class K_Vectors
{
public:
    std::vector<ModuleBase::Vector3<double>> kvec_c; /// Cartesian coordinates of k points
    std::vector<ModuleBase::Vector3<double>> kvec_d; /// Direct coordinates of k points

    std::vector<double> wk; /// wk, weight of k points

    std::vector<int> ngk; /// ngk, number of plane waves for each k point
    std::vector<int> isk; /// distinguish spin up and down k points

    int nmp[3]={0};                 /// Number of Monhorst-Pack
    std::vector<int> kl_segids; /// index of kline segment

    /// @brief equal k points to each ibz-kpont, corresponding to a certain symmetry operations. 
    /// dim: [iks_ibz][(isym, kvec_d)]
    std::vector<std::map<int, ModuleBase::Vector3<double>>> kstars;

    bool kc_done = false;
    bool kd_done = false;

    K_Vectors(){};
    ~K_Vectors(){};
    K_Vectors& operator=(const K_Vectors&) = default;
    K_Vectors& operator=(K_Vectors&& rhs) = default;

    Parallel_Kpoints para_k; ///< parallel for kpoints


    /**
     * @brief Set up the k-points for the system.
     *
     * This function sets up the k-points according to the input parameters and symmetry operations.
     * It also treats the spin as another set of k-points.
     *
     * @param symm The symmetry of the system.
     * @param k_file_name The name of the file containing the k-points.
     * @param nspin_in The number of spins.
     * @param reciprocal_vec The reciprocal vector of the system.
     * @param latvec The lattice vector of the system.
     *
     * @return void
     *
     * @note This function will quit with a warning if something goes wrong while reading the KPOINTS file.
     * @note If the optimized lattice type of the reciprocal lattice cannot match the optimized real lattice,
     *       it will output a warning and suggest possible solutions.
     * @note Only available for nspin = 1 or 2 or 4.
     */
    void set(const UnitCell& ucell,
        const ModuleSymmetry::Symmetry& symm,
        const std::string& k_file_name,
        const int& nspin,
        const ModuleBase::Matrix3& reciprocal_vec,
        const ModuleBase::Matrix3& latvec,
        std::ofstream& ofs);

    int get_nks() const
    {
        return this->nks;
    }

    int get_nkstot() const
    {
        return this->nkstot;
    }

    int get_nkstot_full() const
    {
        return this->nkstot_full;
    }

    double get_koffset(const int i) const
    {
        return this->koffset[i];
    }

    int get_k_nkstot() const
    {
        return this->k_nkstot;
    }

    int get_nspin() const
    {
        return this->nspin;
    }

    std::string get_k_kword() const
    {
        return this->k_kword;
    }

    void set_nks(int value)
    {
        this->nks = value;
    }

    void set_nkstot(int value)
    {
        this->nkstot = value;
    }

    void set_nkstot_full(int value)
    {
        this->nkstot_full = value;
    }

    void set_nspin(int value)
    {
        this->nspin = value;
    }

    bool get_is_mp() const
    {
        return is_mp;
    }

    std::vector<int> ik2iktot; ///<[nks] map ik to the global index of k points

    /**
     * @brief Updates the k-points to use the irreducible Brillouin zone (IBZ).
     *
     * This function updates the k-points to use the irreducible Brillouin zone (IBZ) instead of the full Brillouin
     * zone.
     *
     * @return void
     *
     * @note This function should only be called by the master process (MY_RANK == 0).
     * @note This function assumes that the number of k-points in the IBZ (nkstot_ibz) is greater than 0.
     * @note This function updates the total number of k-points (nkstot) to be the number of k-points in the IBZ.
     * @note This function resizes the vector of k-points (kvec_d) and updates its values to be the k-points in the IBZ.
     * @note This function also updates the weights of the k-points (wk) to be the weights in the IBZ.
     * @note After this function is called, the flag kd_done is set to true to indicate that the k-points have been
     * updated, and the flag kc_done is set to false to indicate that the Cartesian coordinates of the k-points need to
     * be recalculated.
     */
    void update_use_ibz(const int& nkstot_ibz,
                        const std::vector<ModuleBase::Vector3<double>>& kvec_d_ibz,
                        const std::vector<double>& wk_ibz);

  private:
    int nks = 0;         ///< number of symmetry-reduced k points in this pool(processor, up+dw)
    int nkstot = 0;      ///< number of symmetry-reduced k points in full k mesh
    int nkstot_full = 0; ///< number of k points before symmetry reduction in full k mesh

    int nspin = 0;
    double koffset[3] = {0.0}; // used only in automatic k-points.
    std::string k_kword;       // LiuXh add 20180619
    int k_nkstot = 0;          // LiuXh add 20180619 // WHAT IS THIS?????
    bool is_mp = false;        // Monkhorst-Pack

    /**
     * @brief Resize the k-point related vectors according to the new k-point number.
     *
     * This function resizes the vectors that store the k-point information,
     * including the Cartesian and Direct coordinates of k-points,
     * the weights of k-points, the index of k-points, and the number of plane waves for each k-point.
     *
     * @param kpoint_number The new number of k-points.
     *
     * @return void
     *
     * @note The memory recording lines are commented out. If you want to track the memory usage,
     *       you can uncomment these lines.
     */
    void renew(const int& kpoint_number);

    // step 1 : generate kpoints

    /**
     * @brief Reads the k-points from a file.
     *
     * This function reads the k-points from a file specified by the filename.
     * It supports both Cartesian and Direct coordinates, and can handle different types of k-points,
     * including Gamma, Monkhorst-Pack, and Line mode. It also supports automatic generation of k-points
     * file if the file does not exist.
     *
     * @param fn The name of the file containing the k-points.
     *
     * @return bool Returns true if the k-points are successfully read from the file,
     *              false otherwise.
     *
     * @note It will generate a k-points file automatically
     *       according to the global variables GAMMA_ONLY_LOCAL and KSPACING.
     * @note If the k-points type is neither Gamma nor Monkhorst-Pack, it will quit with a warning.
     * @note If the k-points type is Line mode and the symmetry flag is 1, it will quit with a warning.
     * @note If the number of k-points is greater than 100000, it will quit with a warning.
     */
    bool read_kpoints(const UnitCell& ucell,
                      const std::string& fn); // return 0: something wrong.

    /**
     * @brief Adds k-points linearly between special points.
     *
     * This function adds k-points linearly between special points in the Brillouin zone.
     * The special points and the number of k-points between them are read from an input file.
     *
     * @param ifk The input file stream from which the special points and the number of k-points between them are read.
     * @param kvec A vector to store the generated k-points.
     *
     * @return void
     *
     * @note The function first reads the number of special points (nks_special) and the number of k-points between them
     * (nkl) from the input file.
     * @note The function then recalculates the total number of k-points (nkstot) based on the number of k-points
     * between the special points.
     * @note The function generates the k-points by linearly interpolating between the special points.
     * @note The function also assigns a segment ID to each k-point to distinguish different k-line segments.
     * @note The function checks that the total number of generated k-points matches the calculated total number of
     * k-points.
     * @note The function checks that the size of the segment ID vector matches the total number of k-points.
     */
    void interpolate_k_between(std::ifstream& ifk, std::vector<ModuleBase::Vector3<double>>& kvec);

    /**
     * @brief Generates k-points using the Monkhorst-Pack scheme.
     *
     * This function generates k-points in the reciprocal space using the Monkhorst-Pack scheme.
     *
     * @param nmp_in the number of k-points in each dimension.
     * @param koffset_in the offset for the k-points in each dimension.
     * @param k_type The type of k-point.  1 means without Gamma point, 0 means with Gamma.
     *
     * @return void
     *
     * @note The function assumes that the k-points are evenly distributed in the reciprocal space.
     * @note The function sets the weight of each k-point to be equal, so that the total weight of all k-points is 1.
     * @note The function sets the flag kd_done to true to indicate that the k-points have been generated.
     */
    void Monkhorst_Pack(const int* nmp_in, const double* koffset_in, const int tipo);

    /**
     * @brief Calculates the coordinate of a k-point using the Monkhorst-Pack scheme.
     *
     * This function calculates the coordinate of a k-point in the reciprocal space using the Monkhorst-Pack scheme.
     * The Monkhorst-Pack scheme is a method for generating k-points in the Brillouin zone.
     *
     * @param k_type The type of k-point. 1 means without Gamma point, 0 means with Gamma.
     * @param offset The offset for the k-point.
     * @param n The index of the k-point in the current dimension.
     * @param dim The total number of k-points in the current dimension.
     *
     * @return double Returns the coordinate of the k-point.
     *
     * @note The function assumes that the k-points are evenly distributed in the reciprocal space.
     */
    double Monkhorst_Pack_formula(const int& k_type, const double& offset, const int& n, const int& dim);

    // step 2 : set both kvec and kved; normalize weight

    //    void set_both_kvec(const ModuleBase::Matrix3& G, const ModuleBase::Matrix3& R, std::string& skpt);

    /**
     * @brief Normalizes the weights of the k-points.
     *
     * This function normalizes the weights of the k-points so that their sum is equal to the degeneracy of spin
     * (degspin).
     *
     * @param degspin The degeneracy of spin. This is 1 for non-spin-polarized calculations and 2 for spin-polarized
     * calculations.
     *
     * @return void
     *
     * @note This function should only be called by the master process (MY_RANK == 0).
     * @note The function assumes that the sum of the weights of the k-points is greater than 0.
     * @note The function first normalizes the weights so that their sum is 1, and then scales them by the degeneracy of
     * spin.
     */
    void normalize_wk(const int& degspin);



    // step 4 : *2 kpoints.

    /**
     * @brief Sets up the k-points for spin-up and spin-down calculations.
     *
     * This function sets up the k-points for spin-up and spin-down calculations.
     * If the calculation is spin-polarized (nspin = 2), the number of k-points is doubled.
     * The first half of the k-points correspond to spin-up, and the second half correspond to spin-down.
     * 2 for LSDA
     * 4 for non-collinear
     *
     * @return void
     *
     * @note For non-spin-polarized calculations (nspin = 1 or 4), the function simply sets the spin index of all
     * k-points to 0.
     * @note For spin-polarized calculations (nspin = 2), the function duplicates the k-points and their weights,
     *       sets the spin index of the first half of the k-points to 0 (spin-up), and the spin index of the second half
     * to 1 (spin-down).
     * @note The function also doubles the total number of k-points (nks and nkstot) for spin-polarized calculations.
     * @note The function prints the total number of k-points for spin-polarized calculations.
     */
    void set_kup_and_kdw();

    /**
     * @brief Gets the global index of a k-point.
     * @return this->ik2iktot[ik]
     */
    void cal_ik_global();
#ifdef __MPI
    friend void KVectorUtils::kvec_mpi_k(K_Vectors& kvec);
#endif
};
#endif // KVECT_H