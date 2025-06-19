#ifndef UNITCELL_H
#define UNITCELL_H

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "module_elecstate/magnetism.h"
#include "module_io/output.h"
#include "module_symmetry/symmetry.h"

#ifdef __LCAO
#include "module_basis/module_ao/ORB_read.h"
#include "setup_nonlocal.h"
#endif

// provide the basic information about unitcell.
class UnitCell {
  public:
    Atom* atoms = nullptr;

    bool set_atom_flag = false;                     // added on 2009-3-8 by mohan
    Magnetism magnet;                               // magnetism Yu Liu 2021-07-03
    std::vector<std::vector<double>> atom_mulliken; //[nat][nspin]
    int n_mag_at = 0;

    Lattice lat;
    std::string& Coordinate = lat.Coordinate;
    std::string& latName = lat.latName;
    double& lat0 = lat.lat0;
    double& lat0_angstrom = lat.lat0_angstrom;
    double& tpiba = lat.tpiba;
    double& tpiba2 = lat.tpiba2;
    double& omega = lat.omega;
    int*& lc = lat.lc;

    ModuleBase::Matrix3& latvec = lat.latvec;
    ModuleBase::Vector3<double>&a1 = lat.a1, &a2 = lat.a2, &a3 = lat.a3;
    ModuleBase::Vector3<double>& latcenter = lat.latcenter;
    ModuleBase::Matrix3& latvec_supercell = lat.latvec_supercell;
    ModuleBase::Matrix3& G = lat.G;
    ModuleBase::Matrix3& GT = lat.GT;
    ModuleBase::Matrix3& GGT = lat.GGT;
    ModuleBase::Matrix3& invGGT = lat.invGGT;

    Statistics st;
    int& ntype = st.ntype;
    int& nat = st.nat;
    int*& iat2it = st.iat2it;
    int*& iat2ia = st.iat2ia;
    int*& iwt2iat = st.iwt2iat;
    int*& iwt2iw = st.iwt2iw;
    ModuleBase::IntArray& itia2iat = st.itia2iat;
    int& namax = st.namax;
    int& nwmax = st.nwmax;

    ModuleSymmetry::Symmetry symm;

    // ========================================================
    // iat2iwt is the atom index iat to the first global index for orbital of
    // this atom the size of iat2iwt is nat, the value should be
    // sum_{i=0}^{iat-1} atoms[it].nw * npol where the npol is the number of
    // polarizations, 1 for non-magnetic(NSPIN=1 or 2), 2 for magnetic(only
    // NSPIN=4) this part only used for Atomic Orbital based calculation
    // ========================================================
  public:
    // indexing tool for find orbital global index from it,ia,iw
    template <typename Tiait>
    inline Tiait
        itiaiw2iwt(const Tiait& it, const Tiait& ia, const Tiait& iw) const {
        return Tiait(this->iat2iwt[this->itia2iat(it, ia)] + iw);
    }
    // initialize iat2iwt
    void set_iat2iwt(const int& npol_in);
    // get iat2iwt
    inline const int* get_iat2iwt() const { return iat2iwt.data(); }
    // get npol
    inline const int& get_npol() const { return npol; }

  private:
    std::vector<int> iat2iwt; // iat ==> iwt, the first global index for orbital of this atom
    int npol = 1; // number of spin polarizations, initialized in set_iat2iwt
                  // ----------------- END of iat2iwt part -----------------

  public:
    //========================================================
    // indexing tools for ia and it
    // return true if the last out is reset
    //========================================================
    template <typename Tiat, typename Tiait>
    inline bool iat2iait(const Tiat iat, Tiait* ia, Tiait* it) const {
        if (iat >= nat) {
            *ia = 0;
            *it = ntype;
            return false;
        }
        *ia = (Tiait)iat2ia[iat];
        *it = (Tiait)iat2it[iat];
        return true;
    }

    template <typename Tiat, typename Tiait>
    inline bool ijat2iaitjajt(const Tiat ijat,
                              Tiait* ia,
                              Tiait* it,
                              Tiait* ja,
                              Tiait* jt) const {
        Tiat iat = ijat / nat;
        Tiat jat = ijat % nat;
        iat2iait(iat, ia, it);
        iat2iait(jat, ja, jt);
        return true;
    }

    template <typename Tiait>
    inline bool step_it(Tiait* it) const {
        if (++(*it) >= ntype) {
            *it = 0;
            return true;
        }
        return false;
    }

    template <typename Tiait>
    inline bool step_ia(const Tiait it, Tiait* ia) const {
        if (++(*ia) >= atoms[it].na) {
            *ia = 0;
            return true;
        }
        return false;
    }

    template <typename Tiait>
    inline bool step_iait(Tiait* ia, Tiait* it) const {
        if (step_ia(*it, ia)) {
            return step_it(it);
        }
        return false;
    }

    template <typename Tiait>
    inline bool
        step_jajtiait(Tiait* ja, Tiait* jt, Tiait* ia, Tiait* it) const {
        if (step_iait(ja, jt)) {
            return step_iait(ia, it);
        }
        return false;
    }

    // get tau for atom iat
    inline const ModuleBase::Vector3<double>& get_tau(const int& iat) const {
        return atoms[iat2it[iat]].tau[iat2ia[iat]];
    }

    // calculate vector between two atoms with R cell
    inline const ModuleBase::Vector3<double>
        cal_dtau(const int& iat1,
                 const int& iat2,
                 const ModuleBase::Vector3<int>& R) const {
        return get_tau(iat2) + double(R.x) * a1 + double(R.y) * a2
               + double(R.z) * a3 - get_tau(iat1);
    }

    // LiuXh add 20180515
    ModuleBase::Matrix3 G0;
    ModuleBase::Matrix3 GT0;
    ModuleBase::Matrix3 GGT0;
    ModuleBase::Matrix3 invGGT0;

    // I'm doing a bad thing here! Will change later
    bool ionic_position_updated
        = false; // whether the ionic position has been updated
    bool cell_parameter_updated
        = false; // whether the cell parameters are updated

    //============================================================
    // meshx : max number of mesh point in pseudopotential file
    // natomwfc : number of starting wavefunctions
    // lmax  : Max L used for localized orbital.
    // nmax  : Max N used for localized orbital.
    // lmax_ppwf : Max L of pseudo wave functinos
    // nelec : total number of electrons
    // lmaxmax : revert from INPUT
    //============================================================
    int meshx = 0;
    int natomwfc = 0;
    int lmax = 0;
    int nmax = 0;
    int nmax_total = 0; // mohan add 2009-09-10
    int lmax_ppwf = 0;
    int lmaxmax = 0;   // liuyu 2021-07-04
    bool init_vel = false; // liuyu 2021-07-15
                       // double nelec;

  private:
    ModuleBase::Matrix3 stress; // calculate stress on the cell

  public:
    UnitCell();
    ~UnitCell();
    void print_cell(std::ofstream& ofs) const;

    std::vector<double>      atom_mass;
    std::vector<std::string> atom_label;
    std::vector<std::string> pseudo_fn;
    std::vector<std::string> pseudo_type;

    std::vector<std::string> orbital_fn;  // filenames of orbitals, liuyu add 2022-10-19
    std::string  descriptor_file; // filenames of descriptor_file, liuyu add 2023-04-06

    void set_iat2itia();

    void setup_cell(const std::string& fn, std::ofstream& log);

#ifdef __LCAO
    InfoNonlocal infoNL; // store nonlocal information of lcao, added by zhengdy
                         // 2021-09-07
#endif

    // for constrained vc-relaxation where type of lattice
    // is fixed, adjust the lattice vectors

    //================================================================
    // cal_natomwfc : calculate total number of atomic wavefunctions
    // cal_nwfc     : calculate total number of local basis and lmax
    // cal_meshx	: calculate max number of mesh points in pp file
    //================================================================
    bool if_atoms_can_move() const;
    bool if_cell_can_change() const;
    void setup(const std::string& latname_in,
               const int& ntype_in,
               const int& lmaxmax_in,
               const bool& init_vel_in,
               const std::string& fixed_axes_in);

    /// @brief check consistency between two atom labels from STRU and pseudo or
    /// orb file
    void compare_atom_labels(const std::string &label1, const std::string &label2);
    /// @brief get atomCounts, which is a map from element type to atom number
    std::map<int, int> get_atom_Counts() const;
    /// @brief get orbitalCounts, which is a map from element type to orbital
    /// number
    std::map<int, int> get_orbital_Counts() const;
    /// @brief get lnchiCounts, which is a map from element type to the l:nchi
    /// map
    std::map<int, std::map<int, int>> get_lnchi_Counts() const;
    /// these are newly added functions, the three above functions are
    /// deprecated and will be removed in the future
    /// @brief get atom labels
    std::vector<std::string> get_atomLabels() const;
    /// @brief get atomCounts, which is a vector of element type with atom
    /// number
    std::vector<int> get_atomCounts() const;
    /// @brief get lnchiCounts, which is a vector of element type with the
    /// l:nchi vector
    std::vector<std::vector<int>> get_lnchiCounts() const;
    /// @brief get target magnetic moment for deltaspin
    std::vector<ModuleBase::Vector3<double>> get_target_mag() const;
    /// @brief get lagrange multiplier for deltaspin
    std::vector<ModuleBase::Vector3<double>> get_lambda() const;
    /// @brief get constrain for deltaspin
    std::vector<ModuleBase::Vector3<int>> get_constrain() const;
};

#endif // unitcell class
