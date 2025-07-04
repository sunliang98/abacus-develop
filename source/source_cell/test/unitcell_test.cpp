#include "gmock/gmock.h"
#include "gtest/gtest.h"
#define private public
#include "module_parameter/parameter.h"
#undef private
#include "source_estate/cal_ux.h"
#include "source_estate/read_orb.h"
#include "source_estate/read_pseudo.h"
#include "source_cell/read_stru.h"
#include "source_cell/print_cell.h"
#include "memory"
#include "source_cell/read_stru.h"
#include "source_base/global_variable.h"
#include "source_base/mathzone.h"
#include "prepare_unitcell.h"
#include "source_cell/update_cell.h"
#include "source_cell/read_stru.h"
#include <streambuf>
#include <valarray>
#include <vector>

#ifdef __LCAO
#include "source_basis/module_ao/ORB_read.h"
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
LCAO_Orbitals::LCAO_Orbitals()
{
}
LCAO_Orbitals::~LCAO_Orbitals()
{
}
#endif
Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}

/************************************************
 *  unit test of class UnitCell
 ***********************************************/

/**
 * - Tested Functions:
 *   - Constructor:
 *     - UnitCell() and ~UnitCell()
 *   - Setup:
 *     - setup(): to set latname, ntype, lmaxmax, init_vel, and lc
 *     - if_cell_can_change(): judge if any lattice vector can change
 *   - SetupWarningQuit1:
 *     - setup(): deliver warning: "there are bugs in the old implementation;
 *         set relax_new to be 1 for fixed_volume relaxation"
 *   - SetupWarningQuit2:
 *     - setup(): deliver warning: "set relax_new to be 1 for fixed_shape relaxation"
 *   - RemakeCell
 *     - remake_cell(): rebuild cell according to its latName
 *   - RemakeCellWarnings
 *     - remake_cell(): deliver warnings when find wrong latname or cos12
 *   - JudgeParallel
 *     - judge_parallel: judge if two vectors a[3] and Vector3<double> b are parallel
 *   - Index
 *     - set_iat2iait(): set index relations in two arrays of Unitcell: iat2it[nat], iat2ia[nat]
 *     - iat2iait(): depends on the above function, but can find both ia & it from iat
 *     - ijat2iaitjajt(): find ia, it, ja, jt from ijat (ijat_max = nat*nat)
 *         which collapses it, ia, jt, ja loop into a single loop
 *     - step_ia(): periodically set ia to 0 when ia reaches atom[it].na - 1
 *     - step_it(): periodically set it to 0 when it reaches ntype -1
 *     - step_iait(): return true only the above two conditions are true
 *     - step_jajtiait(): return ture only two of the above function (for i and j) are true
 *   - GetAtomCounts
 *     - get_atomCounts(): get atomCounts, which is a map from atom type to atom number
 *   - GetOrbitalCounts
 *     - get_orbitalCounts(): get orbitalCounts, which is a map from atom type to orbital number
 *   - CheckDTau
 *     - check_dtau(): move all atomic coordinates into the first unitcell, i.e. in between [0,1)
 *   - CheckTau
 *     - check_tau(): check if any "two atoms are too close"
 *   - SelectiveDynamics
 *     - if_atoms_can_move():it is true if any coordinates of any atom can move, i.e. mbl = 1
 *   - PeriodicBoundaryAdjustment
 *     - periodic_boundary_adjustment(): move atoms inside the unitcell after relaxation
 *   - PrintCell
 *     - print_cell(ofs): print basic cell info into ofs
 *   - PrintSTRU
 *     - print_stru_file(): print STRU file of ABACUS
 *   - PrintTauDirect
 *   - PrintTauCartesian
 *     - print_tau(): print atomic coordinates, magmom and initial velocities
 *   - PrintUnitcellPseudo
 *     - Actually an integrated function to call UnitCell::print_cell and Atom::print_Atom
 *   - UpdateVel
 *     - update_vel(const ModuleBase::Vector3<double>* vel_in)
 *   - CalUx
 *     - cal_ux(UnitCell& ucell): calculate magnetic moments of cell
 *   - ReadOrbFile
 *     - read_orb_file(): read header part of orbital file
 *   - ReadOrbFileWarning
 *     - read_orb_file(): ABACUS Cannot find the ORBITAL file
 *   - ReadAtomSpecies
 *     - read_atom_species(): a successful case
 *   - ReadAtomSpeciesWarning1
 *     - read_atom_species(): unrecognized pseudopotential type.
 *   - ReadAtomSpeciesWarning2
 *     - read_atom_species(): lat0<=0.0
 *   - ReadAtomSpeciesWarning3
 *     - read_atom_species(): do not use LATTICE_PARAMETERS without explicit specification of lattice type
 *   - ReadAtomSpeciesWarning4
 *     - read_atom_species():do not use LATTICE_VECTORS along with explicit specification of lattice type
 *   - ReadAtomSpeciesWarning5
 *     - read_atom_species():latname not supported
 *   - ReadAtomSpeciesLatName
 *     - read_atom_species(): various latname
 *   - ReadAtomPositionsS1
 *     - read_atom_positions(): spin 1 case
 *   - ReadAtomPositionsS2
 *     - read_atom_positions(): spin 2 case
 *   - ReadAtomPositionsS4Noncolin
 *     - read_atom_positions(): spin 4 noncolinear case
 *   - ReadAtomPositionsS4Colin
 *     - read_atom_positions(): spin 4 colinear case
 *   - ReadAtomPositionsC
 *     - read_atom_positions(): Cartesian coordinates
 *   - ReadAtomPositionsCA
 *     - read_atom_positions(): Cartesian_angstrom coordinates
 *   - ReadAtomPositionsCACXY
 *     - read_atom_positions(): Cartesian_angstrom_center_xy coordinates
 *   - ReadAtomPositionsCACXZ
 *     - read_atom_positions(): Cartesian_angstrom_center_xz coordinates
 *   - ReadAtomPositionsCACXYZ
 *     - read_atom_positions(): Cartesian_angstrom_center_xyz coordinates
 *   - ReadAtomPositionsCAU
 *     - read_atom_positions(): Cartesian_au coordinates
 *   - ReadAtomPositionsWarning1
 *     - read_atom_positions(): unknown type of coordinates
 *   - ReadAtomPositionsWarning2
 *     - read_atom_positions(): atomic label inconsistency between ATOM_POSITIONS
 *                              and ATOM_SPECIES
 *   - ReadAtomPositionsWarning3
 *     - read_atom_positions(): warning :  atom number < 0
 *   - ReadAtomPositionsWarning4
 *     - read_atom_positions(): mismatch in atom number for atom type
 *   - ReadAtomPositionsWarning5
 *     - read_atom_positions(): no atoms can move in MD simulations!
 */

// mock function
#ifdef __LCAO
void LCAO_Orbitals::bcast_files(const int& ntype_in, const int& my_rank)
{
    return;
}
#endif

class UcellTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell{new UnitCell};
    std::string output;
};

using UcellDeathTest = UcellTest;

TEST_F(UcellTest, Constructor)
{
    EXPECT_EQ(ucell->Coordinate, "Direct");
    EXPECT_EQ(ucell->latName, "none");
    EXPECT_DOUBLE_EQ(ucell->lat0, 0.0);
    EXPECT_DOUBLE_EQ(ucell->lat0_angstrom, 0.0);
    EXPECT_EQ(ucell->ntype, 0);
    EXPECT_EQ(ucell->nat, 0);
    EXPECT_EQ(ucell->namax, 0);
    EXPECT_EQ(ucell->nwmax, 0);
    EXPECT_EQ(ucell->iat2it, nullptr);
    EXPECT_EQ(ucell->iat2ia, nullptr);
    EXPECT_EQ(ucell->iwt2iat, nullptr);
    EXPECT_EQ(ucell->iwt2iw, nullptr);
    EXPECT_DOUBLE_EQ(ucell->tpiba, 0.0);
    EXPECT_DOUBLE_EQ(ucell->tpiba2, 0.0);
    EXPECT_DOUBLE_EQ(ucell->omega, 0.0);
    EXPECT_FALSE(ucell->set_atom_flag);
}

TEST_F(UcellTest, Setup)
{
    std::string latname_in = "bcc";
    int ntype_in = 1;
    int lmaxmax_in = 2;
    bool init_vel_in = false;
    std::vector<std::string> fixed_axes_in = {"None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc", "abc"};
    PARAM.input.relax_new = true;
    for (int i = 0; i < fixed_axes_in.size(); ++i)
    {
        ucell->setup(latname_in, ntype_in, lmaxmax_in, init_vel_in, fixed_axes_in[i]);
        EXPECT_EQ(ucell->latName, latname_in);
        EXPECT_EQ(ucell->ntype, ntype_in);
        EXPECT_EQ(ucell->lmaxmax, lmaxmax_in);
        EXPECT_EQ(ucell->init_vel, init_vel_in);
        if (fixed_axes_in[i] == "None" || fixed_axes_in[i] == "volume" || fixed_axes_in[i] == "shape")
        {
            EXPECT_EQ(ucell->lc[0], 1);
            EXPECT_EQ(ucell->lc[1], 1);
            EXPECT_EQ(ucell->lc[2], 1);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "a")
        {
            EXPECT_EQ(ucell->lc[0], 0);
            EXPECT_EQ(ucell->lc[1], 1);
            EXPECT_EQ(ucell->lc[2], 1);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "b")
        {
            EXPECT_EQ(ucell->lc[0], 1);
            EXPECT_EQ(ucell->lc[1], 0);
            EXPECT_EQ(ucell->lc[2], 1);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "c")
        {
            EXPECT_EQ(ucell->lc[0], 1);
            EXPECT_EQ(ucell->lc[1], 1);
            EXPECT_EQ(ucell->lc[2], 0);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "ab")
        {
            EXPECT_EQ(ucell->lc[0], 0);
            EXPECT_EQ(ucell->lc[1], 0);
            EXPECT_EQ(ucell->lc[2], 1);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "ac")
        {
            EXPECT_EQ(ucell->lc[0], 0);
            EXPECT_EQ(ucell->lc[1], 1);
            EXPECT_EQ(ucell->lc[2], 0);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "bc")
        {
            EXPECT_EQ(ucell->lc[0], 1);
            EXPECT_EQ(ucell->lc[1], 0);
            EXPECT_EQ(ucell->lc[2], 0);
            EXPECT_TRUE(ucell->if_cell_can_change());
        }
        else if (fixed_axes_in[i] == "abc")
        {
            EXPECT_EQ(ucell->lc[0], 0);
            EXPECT_EQ(ucell->lc[1], 0);
            EXPECT_EQ(ucell->lc[2], 0);
            EXPECT_FALSE(ucell->if_cell_can_change());
        }
    }
}

TEST_F(UcellDeathTest, SetupWarningQuit1)
{
    std::string latname_in = "bcc";
    int ntype_in = 1;
    int lmaxmax_in = 2;
    bool init_vel_in = false;
    PARAM.input.relax_new = false;
    std::string fixed_axes_in = "volume";
    testing::internal::CaptureStdout();
    EXPECT_EXIT(ucell->setup(latname_in, ntype_in, lmaxmax_in, init_vel_in, fixed_axes_in),
                ::testing::ExitedWithCode(1),
                "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,
                testing::HasSubstr(
                    "there are bugs in the old implementation; set relax_new to be 1 for fixed_volume relaxation"));
}

TEST_F(UcellDeathTest, SetupWarningQuit2)
{
    std::string latname_in = "bcc";
    int ntype_in = 1;
    int lmaxmax_in = 2;
    bool init_vel_in = false;
    PARAM.input.relax_new = false;
    std::string fixed_axes_in = "shape";
    testing::internal::CaptureStdout();
    EXPECT_EXIT(ucell->setup(latname_in, ntype_in, lmaxmax_in, init_vel_in, fixed_axes_in),
                ::testing::ExitedWithCode(1),
                "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("set relax_new to be 1 for fixed_shape relaxation"));
}

TEST_F(UcellDeathTest, CompareAatomLabel)
{
    std::string stru_label[]
        = {"Ag", "Ag", "Ag", "47", "47", "47", "Silver", "Silver", "Silver", "Ag", "Ag", "Ag", "Ag_empty"};
    std::string pseudo_label[]
        = {"Ag", "47", "Silver", "Ag", "47", "Silver", "Ag", "47", "Silver", "Ag1", "ag", "ag_locpsp", "Ag"};
    for (int it = 0; it < 12; it++)
    {
        ucell->compare_atom_labels(stru_label[it], pseudo_label[it]);
    }
    stru_label[0] = "Fe";
    pseudo_label[0] = "O";
    std::string atom_label_in_orbtial = "atom label in orbital file ";
    std::string mismatch_with_pseudo = " mismatch with pseudo file of ";
    testing::internal::CaptureStdout();
    EXPECT_EXIT(ucell->compare_atom_labels(stru_label[0], pseudo_label[0]), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,
                testing::HasSubstr(atom_label_in_orbtial + stru_label[0] + mismatch_with_pseudo + pseudo_label[0]));
}

TEST_F(UcellTest, RemakeCell)
{
    std::vector<std::string> latname_in = {"sc",
                                           "fcc",
                                           "bcc",
                                           "hexagonal",
                                           "trigonal",
                                           "st",
                                           "bct",
                                           "so",
                                           "baco",
                                           "fco",
                                           "bco",
                                           "sm",
                                           "bacm",
                                           "triclinic"};
    for (int i = 0; i < latname_in.size(); ++i)
    {
        ucell->latvec.e11 = 10.0;
        ucell->latvec.e12 = 0.00;
        ucell->latvec.e13 = 0.00;
        ucell->latvec.e21 = 0.00;
        ucell->latvec.e22 = 10.0;
        ucell->latvec.e23 = 0.00;
        ucell->latvec.e31 = 0.00;
        ucell->latvec.e32 = 0.00;
        ucell->latvec.e33 = 10.0;
        ucell->latName = latname_in[i];
        unitcell::remake_cell(ucell->lat);
        if (latname_in[i] == "sc")
        {
            double celldm
                = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, celldm);
        }
        else if (latname_in[i] == "fcc")
        {
            double celldm = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2))
                            / std::sqrt(2.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, -celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, -celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, 0.0);
        }
        else if (latname_in[i] == "bcc")
        {
            double celldm = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2))
                            / std::sqrt(3.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, -celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, -celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, -celldm);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, celldm);
        }
        else if (latname_in[i] == "hexagonal")
        {
            double celldm1
                = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double celldm3
                = std::sqrt(pow(ucell->latvec.e31, 2) + pow(ucell->latvec.e32, 2) + pow(ucell->latvec.e33, 2));
            double mathfoo = sqrt(3.0) / 2.0;
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, celldm1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, -0.5 * celldm1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, celldm1 * mathfoo);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, celldm3);
        }
        else if (latname_in[i] == "trigonal")
        {
            double a1 = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double a2 = std::sqrt(pow(ucell->latvec.e21, 2) + pow(ucell->latvec.e22, 2) + pow(ucell->latvec.e23, 2));
            double a1da2 = (ucell->latvec.e11 * ucell->latvec.e21 + ucell->latvec.e12 * ucell->latvec.e22
                            + ucell->latvec.e13 * ucell->latvec.e23);
            double cosgamma = a1da2 / (a1 * a2);
            double tx = std::sqrt((1.0 - cosgamma) / 2.0);
            double ty = std::sqrt((1.0 - cosgamma) / 6.0);
            double tz = std::sqrt((1.0 + 2.0 * cosgamma) / 3.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, a1 * tx);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, -a1 * ty);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, a1 * tz);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, 2.0 * a1 * ty);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, a1 * tz);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, -a1 * tx);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, -a1 * ty);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, a1 * tz);
        }
        else if (latname_in[i] == "st")
        {
            double a1 = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double a3 = std::sqrt(pow(ucell->latvec.e31, 2) + pow(ucell->latvec.e32, 2) + pow(ucell->latvec.e33, 2));
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, a1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, a1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, a3);
        }
        else if (latname_in[i] == "bct")
        {
            double d1 = std::abs(ucell->latvec.e11);
            double d2 = std::abs(ucell->latvec.e13);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, d2);
        }
        else if (latname_in[i] == "so")
        {
            double a1 = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double a2 = std::sqrt(pow(ucell->latvec.e21, 2) + pow(ucell->latvec.e22, 2) + pow(ucell->latvec.e23, 2));
            double a3 = std::sqrt(pow(ucell->latvec.e31, 2) + pow(ucell->latvec.e32, 2) + pow(ucell->latvec.e33, 2));
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, a1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, a2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, a3);
        }
        else if (latname_in[i] == "baco")
        {
            double d1 = std::abs(ucell->latvec.e11);
            double d2 = std::abs(ucell->latvec.e22);
            double d3 = std::abs(ucell->latvec.e33);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, d3);
        }
        else if (latname_in[i] == "fco")
        {
            double d1 = std::abs(ucell->latvec.e11);
            double d2 = std::abs(ucell->latvec.e22);
            double d3 = std::abs(ucell->latvec.e33);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, d3);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, d3);
        }
        else if (latname_in[i] == "bco")
        {
            double d1 = std::abs(ucell->latvec.e11);
            double d2 = std::abs(ucell->latvec.e22);
            double d3 = std::abs(ucell->latvec.e33);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, d3);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, d3);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, -d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, -d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, d3);
        }
        else if (latname_in[i] == "sm")
        {
            double a1 = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double a2 = std::sqrt(pow(ucell->latvec.e21, 2) + pow(ucell->latvec.e22, 2) + pow(ucell->latvec.e23, 2));
            double a3 = std::sqrt(pow(ucell->latvec.e31, 2) + pow(ucell->latvec.e32, 2) + pow(ucell->latvec.e33, 2));
            double a1da2 = (ucell->latvec.e11 * ucell->latvec.e21 + ucell->latvec.e12 * ucell->latvec.e22
                            + ucell->latvec.e13 * ucell->latvec.e23);
            double cosgamma = a1da2 / (a1 * a2);
            double d1 = a2 * cosgamma;
            double d2 = a2 * std::sqrt(1.0 - cosgamma * cosgamma);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, a1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, d2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, a3);
        }
        else if (latname_in[i] == "bacm")
        {
            double d1 = std::abs(ucell->latvec.e11);
            double a2 = std::sqrt(pow(ucell->latvec.e21, 2) + pow(ucell->latvec.e22, 2) + pow(ucell->latvec.e23, 2));
            double d3 = std::abs(ucell->latvec.e13);
            double cosgamma = ucell->latvec.e21 / a2;
            double f1 = a2 * cosgamma;
            double f2 = a2 * std::sqrt(1.0 - cosgamma * cosgamma);
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, -d3);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, f1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, f2);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, d1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, d3);
        }
        else if (latname_in[i] == "triclinic")
        {
            double a1 = std::sqrt(pow(ucell->latvec.e11, 2) + pow(ucell->latvec.e12, 2) + pow(ucell->latvec.e13, 2));
            double a2 = std::sqrt(pow(ucell->latvec.e21, 2) + pow(ucell->latvec.e22, 2) + pow(ucell->latvec.e23, 2));
            double a3 = std::sqrt(pow(ucell->latvec.e31, 2) + pow(ucell->latvec.e32, 2) + pow(ucell->latvec.e33, 2));
            double a1da2 = (ucell->latvec.e11 * ucell->latvec.e21 + ucell->latvec.e12 * ucell->latvec.e22
                            + ucell->latvec.e13 * ucell->latvec.e23);
            double a1da3 = (ucell->latvec.e11 * ucell->latvec.e31 + ucell->latvec.e12 * ucell->latvec.e32
                            + ucell->latvec.e13 * ucell->latvec.e33);
            double a2da3 = (ucell->latvec.e21 * ucell->latvec.e31 + ucell->latvec.e22 * ucell->latvec.e32
                            + ucell->latvec.e23 * ucell->latvec.e33);
            double cosgamma = a1da2 / a1 / a2;
            double singamma = std::sqrt(1.0 - cosgamma * cosgamma);
            double cosbeta = a1da3 / a1 / a3;
            double cosalpha = a2da3 / a2 / a3;
            double d1 = std::sqrt(1.0 + 2.0 * cosgamma * cosbeta * cosalpha - cosgamma * cosgamma - cosbeta * cosbeta
                                  - cosalpha * cosalpha)
                        / singamma;
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, a1);
            EXPECT_DOUBLE_EQ(ucell->latvec.e12, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e13, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e21, a2 * cosgamma);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, a2 * singamma);
            EXPECT_DOUBLE_EQ(ucell->latvec.e23, 0.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e31, a3 * cosbeta);
            EXPECT_DOUBLE_EQ(ucell->latvec.e32, a3 * (cosalpha - cosbeta * cosgamma) / singamma);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, a3 * d1);
        }
    }
}

TEST_F(UcellDeathTest, RemakeCellWarnings)
{
    std::vector<std::string> latname_in = {"none", "trigonal", "bacm", "triclinic", "arbitrary"};
    for (int i = 0; i < latname_in.size(); ++i)
    {
        ucell->latvec.e11 = 10.0;
        ucell->latvec.e12 = 0.00;
        ucell->latvec.e13 = 0.00;
        ucell->latvec.e21 = 10.0;
        ucell->latvec.e22 = 0.00;
        ucell->latvec.e23 = 0.00;
        ucell->latvec.e31 = 0.00;
        ucell->latvec.e32 = 0.00;
        ucell->latvec.e33 = 10.0;
        ucell->latName = latname_in[i];
        testing::internal::CaptureStdout();
        EXPECT_EXIT(unitcell::remake_cell(ucell->lat), ::testing::ExitedWithCode(1), "");
        std::string output = testing::internal::GetCapturedStdout();
        if (latname_in[i] == "none")
        {
            EXPECT_THAT(output, testing::HasSubstr("to use fixed_ibrav, latname must be provided"));
        }
        else if (latname_in[i] == "trigonal" || latname_in[i] == "bacm" || latname_in[i] == "triclinic")
        {
            EXPECT_THAT(output, testing::HasSubstr("wrong cos12!"));
        }
        else
        {
            EXPECT_THAT(output, testing::HasSubstr("latname not supported!"));
        }
    }
}

TEST_F(UcellTest, JudgeParallel)
{
    ModuleBase::Vector3<double> b(1.0, 1.0, 1.0);
    double a[3] = {1.0, 1.0, 1.0};
    EXPECT_TRUE(elecstate::judge_parallel(a, b));
}

TEST_F(UcellTest, Index)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    // test set_iat2itia
    ucell->set_iat2itia();
    int iat = 0;
    for (int it = 0; it < utp.natom.size(); ++it)
    {
        for (int ia = 0; ia < utp.natom[it]; ++ia)
        {
            EXPECT_EQ(ucell->iat2it[iat], it);
            EXPECT_EQ(ucell->iat2ia[iat], ia);
            // test iat2iait
            int ia_beg, it_beg;
            ucell->iat2iait(iat, &ia_beg, &it_beg);
            EXPECT_EQ(it_beg, it);
            EXPECT_EQ(ia_beg, ia);
            ++iat;
        }
    }
    // test iat2iait: case of (iat >= nat)
    int ia_beg2;
    int it_beg2;
    long long iat2 = ucell->nat + 1;
    EXPECT_FALSE(ucell->iat2iait(iat2, &ia_beg2, &it_beg2));
    // test ijat2iaitjajt, step_jajtiait, step_iat, step_ia, step_it
    int ia_test;
    int it_test;
    int ja_test;
    int jt_test;
    int ia_test2 = 0;
    int it_test2 = 0;
    int ja_test2 = 0;
    int jt_test2 = 0;
    long long ijat = 0;
    for (int it = 0; it < utp.natom.size(); ++it)
    {
        for (int ia = 0; ia < utp.natom[it]; ++ia)
        {
            for (int jt = 0; jt < utp.natom.size(); ++jt)
            {
                for (int ja = 0; ja < utp.natom[jt]; ++ja)
                {
                    ucell->ijat2iaitjajt(ijat, &ia_test, &it_test, &ja_test, &jt_test);
                    EXPECT_EQ(ia_test, ia);
                    EXPECT_EQ(it_test, it);
                    EXPECT_EQ(ja_test, ja);
                    EXPECT_EQ(jt_test, jt);
                    ++ijat;
                    if (it_test == utp.natom.size() - 1 && ia_test == utp.natom[it] - 1
                        && jt_test == utp.natom.size() - 1 && ja_test == utp.natom[jt] - 1)
                    {
                        EXPECT_TRUE(ucell->step_jajtiait(&ja_test, &jt_test, &ia_test, &it_test));
                    }
                    else
                    {
                        EXPECT_FALSE(ucell->step_jajtiait(&ja_test, &jt_test, &ia_test, &it_test));
                    }
                }
            }
        }
    }
}

TEST_F(UcellTest, GetAtomCounts)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    // test set_iat2itia
    ucell->set_iat2itia();
    std::map<int, int> atomCounts = ucell->get_atom_Counts();
    EXPECT_EQ(atomCounts[0], 1);
    EXPECT_EQ(atomCounts[1], 2);
    /// atomCounts as vector
    std::vector<int> atomCounts2 = ucell->get_atomCounts();
    EXPECT_EQ(atomCounts2[0], 1);
    EXPECT_EQ(atomCounts2[1], 2);
}

TEST_F(UcellTest, GetOrbitalCounts)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    // test set_iat2itia
    ucell->set_iat2itia();
    std::map<int, int> orbitalCounts = ucell->get_orbital_Counts();
    EXPECT_EQ(orbitalCounts[0], 9);
    EXPECT_EQ(orbitalCounts[1], 9);
}

TEST_F(UcellTest, GetLnchiCounts)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    // test set_iat2itia
    ucell->set_iat2itia();
    std::map<int, std::map<int, int>> LnchiCounts = ucell->get_lnchi_Counts();
    EXPECT_EQ(LnchiCounts[0][0], 1);
    EXPECT_EQ(LnchiCounts[0][1], 1);
    EXPECT_EQ(LnchiCounts[0][2], 1);
    EXPECT_EQ(LnchiCounts[1][0], 1);
    EXPECT_EQ(LnchiCounts[1][1], 1);
    EXPECT_EQ(LnchiCounts[1][2], 1);
    /// LnchiCounts as vector
    std::vector<std::vector<int>> LnchiCounts2 = ucell->get_lnchiCounts();
    EXPECT_EQ(LnchiCounts2[0][0], 1);
    EXPECT_EQ(LnchiCounts2[0][1], 1);
    EXPECT_EQ(LnchiCounts2[0][2], 1);
    EXPECT_EQ(LnchiCounts2[1][0], 1);
    EXPECT_EQ(LnchiCounts2[1][1], 1);
    EXPECT_EQ(LnchiCounts2[1][2], 1);
}

TEST_F(UcellTest, CheckDTau)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-CheckDTau"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    unitcell::check_dtau(ucell->atoms,ucell->ntype, ucell->lat0, ucell->latvec);
    for (int it = 0; it < utp.natom.size(); ++it)
    {
        for (int ia = 0; ia < utp.natom[it]; ++ia)
        {
            EXPECT_GE(ucell->atoms[it].taud[ia].x, 0);
            EXPECT_GE(ucell->atoms[it].taud[ia].y, 0);
            EXPECT_GE(ucell->atoms[it].taud[ia].z, 0);
            EXPECT_LT(ucell->atoms[it].taud[ia].x, 1);
            EXPECT_LT(ucell->atoms[it].taud[ia].y, 1);
            EXPECT_LT(ucell->atoms[it].taud[ia].z, 1);
        }
    }
}

TEST_F(UcellTest, CheckTauFalse)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-CheckTau"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    GlobalV::ofs_warning.open("checktau_warning");
    unitcell::check_tau(ucell->atoms ,ucell->ntype, ucell->lat0);
    GlobalV::ofs_warning.close();
    std::ifstream ifs;
    ifs.open("checktau_warning");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("two atoms are too close!"));
    ifs.close();
    remove("checktau_warning");
}

TEST_F(UcellTest, CheckTauTrue)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-CheckTau"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    GlobalV::ofs_warning.open("checktau_warning");
    int atom=0;
    //cause the ucell->lat0 is 0.5,if the type of the check_tau has 
    //an int type,it will set to zero,and it will not pass the unittest
    ucell->lat0=0.5;
    ucell->nat=3;
    for (int it=0;it<ucell->ntype;it++)
    {
        for(int ia=0; ia<ucell->atoms[it].na; ++ia)
        {
            
            for (int i=0;i<3;i++)
            {
                ucell->atoms[it].tau[ia][i]=((atom+i)/(ucell->nat*3.0));
            }
            atom+=3;
        }
    }
    EXPECT_EQ(unitcell::check_tau(ucell->atoms ,ucell->ntype, ucell->lat0),true);
    GlobalV::ofs_warning.close();
}

TEST_F(UcellTest, SelectiveDynamics)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-SD"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    EXPECT_TRUE(ucell->if_atoms_can_move());
}

TEST_F(UcellDeathTest, PeriodicBoundaryAdjustment1)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-PBA"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::periodic_boundary_adjustment(
                ucell->atoms,ucell->latvec,ucell->ntype),
                ::testing::ExitedWithCode(1), "");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("the movement of atom is larger than the length of cell"));
}

TEST_F(UcellTest, PeriodicBoundaryAdjustment2)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    EXPECT_NO_THROW(unitcell::periodic_boundary_adjustment(
                    ucell->atoms,ucell->latvec,ucell->ntype));
}

TEST_F(UcellTest, PrintCell)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    std::ofstream ofs;
    ofs.open("printcell.log");
    ucell->print_cell(ofs);
    ofs.close();
    std::ifstream ifs;
    ifs.open("printcell.log");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("latName = bcc"));
    EXPECT_THAT(str, testing::HasSubstr("ntype = 2"));
    EXPECT_THAT(str, testing::HasSubstr("nat = 3"));
    EXPECT_THAT(str, testing::HasSubstr("GGT :"));
    EXPECT_THAT(str, testing::HasSubstr("omega = 6748.33"));
    remove("printcell.log");
}

TEST_F(UcellTest, PrintUnitcellPseudo)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    PARAM.input.test_pseudo_cell = 1;
    std::string fn = "printcell.log";
    elecstate::print_unitcell_pseudo(fn, *ucell);
    std::ifstream ifs;
    ifs.open("printcell.log");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("latName = bcc"));
    EXPECT_THAT(str, testing::HasSubstr("ntype = 2"));
    EXPECT_THAT(str, testing::HasSubstr("nat = 3"));
    EXPECT_THAT(str, testing::HasSubstr("GGT :"));
    EXPECT_THAT(str, testing::HasSubstr("omega = 6748.33"));
    EXPECT_THAT(str, testing::HasSubstr("label = C"));
    EXPECT_THAT(str, testing::HasSubstr("mass = 12"));
    EXPECT_THAT(str, testing::HasSubstr("atom_position(cartesian) Dimension = 1"));
    EXPECT_THAT(str, testing::HasSubstr("label = H"));
    EXPECT_THAT(str, testing::HasSubstr("mass = 1"));
    EXPECT_THAT(str, testing::HasSubstr("atom_position(cartesian) Dimension = 2"));
    remove("printcell.log");
}

// Comments and suggestions on the refactor of UnitCell class
// the test of this function may relies on a ABACUS STRU parser to fully rationally proceed.
// however the parser is not easy to be ready cause the structure of STRU may need to be
// re-designed for being better-orgnized.
// based on present situation, the unittest can only be feasibly written with substr check,
// but will time and time again raise error if format change even a little bit.
// if allow user to change the precision of quantities printed out, then this unittest cannot
// cover these kind of cases.
// In summmary, there are two cents:
// 1. STRU file needed to be re-designed to be more well-organized
// 2. STRU file parser can therefore be programmed more succinctly
TEST_F(UcellTest, PrintSTRU)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    // Cartesian type of coordinates
    std::string fn = "C1H2_STRU";
    PARAM.input.calculation = "md"; // print velocity in STRU, not needed anymore after refactor of this function

    /**
     * CASE: nspin1|Cartesian|no vel|no mag|no orb|no dpks_desc|rank0
     *
     */
    unitcell::print_stru_file(*ucell,ucell->atoms,ucell->latvec,
                              fn, 1, false, false, false, false, false, 0);
    std::ifstream ifs;
    ifs.open("C1H2_STRU");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_SPECIES"));
    EXPECT_THAT(str, testing::HasSubstr("C  12.0000 C.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("H   1.0000 H.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_CONSTANT"));
    EXPECT_THAT(str, testing::HasSubstr("1.8897261255"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_VECTORS"));
    EXPECT_THAT(str, testing::HasSubstr("10.0000000000        0.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000       10.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000        0.0000000000       10.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_POSITIONS"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian"));
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("1 #number of atoms"));
    EXPECT_THAT(str, testing::HasSubstr("        1.0000000000        1.0000000000        1.0000000000 m 1 1 1"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("2 #number of atoms"));
    EXPECT_THAT(str, testing::HasSubstr("        1.5000000000        1.5000000000        1.5000000000 m 0 0 0"));
    EXPECT_THAT(str, testing::HasSubstr("        0.5000000000        0.5000000000        0.5000000000 m 0 0 1"));
    str.clear();
    ifs.close();
    remove("C1H2_STRU");
    /**
     * CASE: nspin2|Direct|vel|no mag|no orb|no dpks_desc|rank0
     *
     */
    unitcell::print_stru_file(*ucell,ucell->atoms,ucell->latvec,
                            fn, 2, true, true, false, false, false, 0);
    ifs.open("C1H2_STRU");
    str = {(std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()};
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_SPECIES"));
    EXPECT_THAT(str, testing::HasSubstr("C  12.0000 C.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("H   1.0000 H.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_CONSTANT"));
    EXPECT_THAT(str, testing::HasSubstr("1.8897261255"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_VECTORS"));
    EXPECT_THAT(str, testing::HasSubstr("10.0000000000        0.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000       10.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000        0.0000000000       10.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_POSITIONS"));
    EXPECT_THAT(str, testing::HasSubstr("Direct"));
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("1 #number of atoms"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.1000000000        0.1000000000        0.1000000000 m 1 1 1 v        "
                                   "0.1000000000        0.1000000000        0.1000000000"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("2 #number of atoms"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.1500000000        0.1500000000        0.1500000000 m 0 0 0 v        "
                                   "0.1000000000        0.1000000000        0.1000000000"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.0500000000        0.0500000000        0.0500000000 m 0 0 1 v        "
                                   "0.1000000000        0.1000000000        0.1000000000"));
    str.clear();
    ifs.close();
    remove("C1H2_STRU");
    /**
     * CASE: nspin2|Direct|no vel|mag|orb|dpks_desc|rank0
     *
     */
    ucell->descriptor_file = "__unittest_numerical_descriptor__";
    ucell->orbital_fn[0] = "__unittest_orbital_fn_0__";
    ucell->orbital_fn[1] = "__unittest_orbital_fn_1__";
    ucell->atom_mulliken
        = {{-1, 0.5}, {-1, 0.4}, {-1, 0.3}}; // first index is iat, the second is components, starts seems from 1
    unitcell::print_stru_file(*ucell,ucell->atoms,ucell->latvec,
                            fn, 2, true, false, true, true, true, 0);
    ifs.open("C1H2_STRU");
    str = {(std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()};
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_SPECIES"));
    EXPECT_THAT(str, testing::HasSubstr("C  12.0000 C.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("H   1.0000 H.upf upf201"));
    EXPECT_THAT(str, testing::HasSubstr("NUMERICAL_ORBITAL"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_orbital_fn_0__"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_orbital_fn_1__"));
    EXPECT_THAT(str, testing::HasSubstr("NUMERICAL_DESCRIPTOR"));
    EXPECT_THAT(str, testing::HasSubstr("__unittest_numerical_descriptor__"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_CONSTANT"));
    EXPECT_THAT(str, testing::HasSubstr("1.8897261255"));
    EXPECT_THAT(str, testing::HasSubstr("LATTICE_VECTORS"));
    EXPECT_THAT(str, testing::HasSubstr("10.0000000000        0.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000       10.0000000000        0.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr(" 0.0000000000        0.0000000000       10.0000000000"));
    EXPECT_THAT(str, testing::HasSubstr("ATOMIC_POSITIONS"));
    EXPECT_THAT(str, testing::HasSubstr("Direct"));
    EXPECT_THAT(str, testing::HasSubstr("C #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("1 #number of atoms"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.1000000000        0.1000000000        0.1000000000 m 1 1 1 mag  0.5000"));
    EXPECT_THAT(str, testing::HasSubstr("H #label"));
    EXPECT_THAT(str, testing::HasSubstr("0.0000   #magnetism"));
    EXPECT_THAT(str, testing::HasSubstr("2 #number of atoms"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.1500000000        0.1500000000        0.1500000000 m 0 0 0 mag  0.4000"));
    EXPECT_THAT(str,
                testing::HasSubstr("        0.0500000000        0.0500000000        0.0500000000 m 0 0 1 mag  0.3000"));
    ifs.close();
    remove("C1H2_STRU");
}

TEST_F(UcellTest, PrintTauDirect)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    EXPECT_EQ(ucell->Coordinate, "Direct");

    // open a file
    std::ofstream ofs("print_tau_direct");
    unitcell::print_tau(ucell->atoms,ucell->Coordinate,ucell->ntype,ucell->lat0,ofs);
    ofs.close();
 
    // readin the data
    std::ifstream ifs;
    ifs.open("print_tau_direct");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("DIRECT COORDINATES"));
    EXPECT_THAT(str, testing::HasSubstr("    C     0.100000000000     0.100000000000     0.100000000000  0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("    H     0.150000000000     0.150000000000     0.150000000000  0.0000")); 
    ifs.close();

    remove("print_tau_direct");
}

TEST_F(UcellTest, PrintTauCartesian)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Cartesian"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    EXPECT_EQ(ucell->Coordinate, "Cartesian");

    // open a file
    std::ofstream ofs("print_tau_Cartesian");
    unitcell::print_tau(ucell->atoms,ucell->Coordinate,ucell->ntype,ucell->lat0,ofs);
    ofs.close();

    // readin the data
    std::ifstream ifs;
    ifs.open("print_tau_Cartesian");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("CARTESIAN COORDINATES"));
    EXPECT_THAT(str, testing::HasSubstr("    C     1.000000000000     1.000000000000     1.000000000000  0.0000"));
    EXPECT_THAT(str, testing::HasSubstr("    H     1.500000000000     1.500000000000     1.500000000000  0.0000"));
    ifs.close();

    // remove the file
    remove("print_tau_Cartesian");
}

TEST_F(UcellTest, UpdateVel)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Index"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    ModuleBase::Vector3<double>* vel_in = new ModuleBase::Vector3<double>[ucell->nat];
    for (int iat = 0; iat < ucell->nat; ++iat)
    {
        vel_in[iat].set(iat * 0.1, iat * 0.1, iat * 0.1);
    }
    unitcell::update_vel(vel_in,ucell->ntype,ucell->nat,ucell->atoms);
    for (int iat = 0; iat < ucell->nat; ++iat)
    {
        EXPECT_DOUBLE_EQ(vel_in[iat].x, 0.1 * iat);
        EXPECT_DOUBLE_EQ(vel_in[iat].y, 0.1 * iat);
        EXPECT_DOUBLE_EQ(vel_in[iat].z, 0.1 * iat);
    }
    delete[] vel_in;
}

TEST_F(UcellTest, CalUx1)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Read"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    ucell->atoms[0].m_loc_[0].set(0, -1, 0);
    ucell->atoms[1].m_loc_[0].set(1, 1, 1);
    ucell->atoms[1].m_loc_[1].set(0, 0, 0);
    PARAM.input.nspin = 4;
    elecstate::cal_ux(*ucell);
    EXPECT_FALSE(ucell->magnet.lsign_);
    EXPECT_DOUBLE_EQ(ucell->magnet.ux_[0], 0);
    EXPECT_DOUBLE_EQ(ucell->magnet.ux_[1], -1);
    EXPECT_DOUBLE_EQ(ucell->magnet.ux_[2], 0);
}

TEST_F(UcellTest, CalUx2)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Read"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    ucell->atoms[0].m_loc_[0].set(0, 0, 0);
    ucell->atoms[1].m_loc_[0].set(1, 1, 1);
    ucell->atoms[1].m_loc_[1].set(0, 0, 0);
    //(0,0,0) is also parallel to (1,1,1)
    PARAM.input.nspin = 4;
    elecstate::cal_ux(*ucell);
    EXPECT_TRUE(ucell->magnet.lsign_);
    EXPECT_NEAR(ucell->magnet.ux_[0], 0.57735, 1e-5);
    EXPECT_NEAR(ucell->magnet.ux_[1], 0.57735, 1e-5);
    EXPECT_NEAR(ucell->magnet.ux_[2], 0.57735, 1e-5);
}

#ifdef __LCAO
TEST_F(UcellTest, ReadOrbFile)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Read"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    std::string orb_file = "./support/C.orb";
    std::ofstream ofs_running;
    ofs_running.open("tmp_readorbfile");
    elecstate::read_orb_file(0, orb_file, ofs_running, &(ucell->atoms[0]));
    ofs_running.close();
    EXPECT_EQ(ucell->atoms[0].nw, 25);
    remove("tmp_readorbfile");
}

TEST_F(UcellDeathTest, ReadOrbFileWarning)
{
    UcellTestPrepare utp = UcellTestLib["C1H2-Read"];
    PARAM.input.relax_new = utp.relax_new;
    ucell = utp.SetUcellInfo();
    std::string orb_file = "./support/CC.orb";
    std::ofstream ofs_running;
    ofs_running.open("tmp_readorbfile");
    testing::internal::CaptureStdout();
    EXPECT_EXIT(elecstate::read_orb_file(0, orb_file, ofs_running, &(ucell->atoms[0])), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("ABACUS Cannot find the ORBITAL file"));
    ofs_running.close();
    remove("tmp_readorbfile");
}
class UcellTestReadStru : public ::testing::Test
{
  protected:
    std::unique_ptr<UnitCell> ucell{new UnitCell};
    std::string output;
  	void SetUp() override
    {
    	ucell->ntype = 2;
        ucell->atom_mass.resize(ucell->ntype);
        ucell->atom_label.resize(ucell->ntype);
        ucell->pseudo_fn.resize(ucell->ntype);
        ucell->pseudo_type.resize(ucell->ntype);
        ucell->orbital_fn.resize(ucell->ntype);
    }
    void TearDown() override
    {
        ucell->orbital_fn.shrink_to_fit();
    }
};

TEST_F(UcellTestReadStru, ReadAtomSpecies)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species.tmp");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running, ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    ofs_running.close();
    ifa.close();
    remove("read_atom_species.tmp");
}

TEST_F(UcellTestReadStru, ReadAtomSpeciesWarning1)
{
    std::string fn = "./support/STRU_MgO_Warning1";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species.txt");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_atom_species(ifa, ofs_running,*ucell), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("unrecognized pseudopotential type."));
    ofs_running.close();
    ifa.close();
    //remove("read_atom_species.txt");
}

TEST_F(UcellTestReadStru, ReadLatticeConstantWarning1)
{
    std::string fn = "./support/STRU_MgO_Warning2";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species1.tmp");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("Lattice constant <= 0.0"));
    ofs_running.close();
    ifa.close();
    remove("read_atom_species1.tmp");
}

TEST_F(UcellTestReadStru, ReadLatticeConstantWarning2)
{
    std::string fn = "./support/STRU_MgO_Warning3";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species.tmp");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,
                testing::HasSubstr("do not use LATTICE_PARAMETERS without explicit specification of lattice type"));
    ofs_running.close();
    ifa.close();
    remove("read_atom_species.tmp");
}

TEST_F(UcellTestReadStru, ReadLatticeConstantWarning3)
{
    std::string fn = "./support/STRU_MgO_Warning4";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species.tmp");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    ucell->latName = "bcc";
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,
                testing::HasSubstr("do not use LATTICE_VECTORS along with explicit specification of lattice type"));
    ofs_running.close();
    ifa.close();
    remove("read_atom_species.tmp");
}

TEST_F(UcellTestReadStru, ReadAtomSpeciesLatName)
{
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    std::vector<std::string> latName_in = {"sc",
                                           "fcc",
                                           "bcc",
                                           "hexagonal",
                                           "trigonal",
                                           "st",
                                           "bct",
                                           "so",
                                           "baco",
                                           "fco",
                                           "bco",
                                           "sm",
                                           "bacm",
                                           "triclinic"};
    for (int i = 0; i < latName_in.size(); ++i)
    {
        std::string fn = "./support/STRU_MgO_LatName";
        std::ifstream ifa(fn.c_str());
        std::ofstream ofs_running;
        ofs_running.open("read_atom_species.tmp");
        ucell->latName = latName_in[i];
        EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
        if (ucell->latName == "sc")
        {
            EXPECT_DOUBLE_EQ(ucell->latvec.e11, 1.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e22, 1.0);
            EXPECT_DOUBLE_EQ(ucell->latvec.e33, 1.0);
        }
        ofs_running.close();
        ifa.close();
        remove("read_atom_species.tmp");
    }
}

TEST_F(UcellDeathTest, ReadAtomSpeciesWarning5)
{
    std::string fn = "./support/STRU_MgO_LatName";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_species.tmp");
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    ucell->latName = "arbitrary";
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("latname not supported"));
    ofs_running.close();
    ifa.close();
    remove("read_atom_species.tmp");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsS1)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsS2)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 2;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsS4Noncolin)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 4;
    PARAM.input.noncolin = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsS4Colin)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 4;
    PARAM.input.noncolin = false;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsC)
{
    std::string fn = "./support/STRU_MgO_c";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCA)
{
    std::string fn = "./support/STRU_MgO_ca";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCACXY)
{
    std::string fn = "./support/STRU_MgO_cacxy";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCACXZ)
{
    std::string fn = "./support/STRU_MgO_cacxz";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCACYZ)
{
    std::string fn = "./support/STRU_MgO_cacyz";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCACXYZ)
{
    std::string fn = "./support/STRU_MgO_cacxyz";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsCAU)
{
    std::string fn = "./support/STRU_MgO_cau";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 1;
    PARAM.input.fixed_atoms = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsAutosetMag)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.nspin = 2;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    for (int it = 0; it < ucell->ntype; it++)
    {
        for (int ia = 0; ia < ucell->atoms[it].na; ia++)
        {
            EXPECT_DOUBLE_EQ(ucell->atoms[it].mag[ia], 1.0);
            EXPECT_DOUBLE_EQ(ucell->atoms[it].m_loc_[ia].x, 1.0);
        }
    }
    // for nspin == 4
    PARAM.input.nspin = 4;
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning);
    for (int it = 0; it < ucell->ntype; it++)
    {
        for (int ia = 0; ia < ucell->atoms[it].na; ia++)
        {
            EXPECT_DOUBLE_EQ(ucell->atoms[it].mag[ia], sqrt(pow(1.0, 2) + pow(1.0, 2) + pow(1.0, 2)));
            EXPECT_DOUBLE_EQ(ucell->atoms[it].m_loc_[ia].x, 1.0);
            EXPECT_DOUBLE_EQ(ucell->atoms[it].m_loc_[ia].y, 1.0);
            EXPECT_DOUBLE_EQ(ucell->atoms[it].m_loc_[ia].z, 1.0);
        }
    }
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsWarning1)
{
    std::string fn = "./support/STRU_MgO_WarningC1";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    EXPECT_NO_THROW(unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning));
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    // check warning file
    std::ifstream ifs_tmp;
    ifs_tmp.open("read_atom_positions.warn");
    std::string str((std::istreambuf_iterator<char>(ifs_tmp)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("There are several options for you:"));
    EXPECT_THAT(str, testing::HasSubstr("Direct"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_au"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom_center_xy"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom_center_xz"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom_center_yz"));
    EXPECT_THAT(str, testing::HasSubstr("Cartesian_angstrom_center_xyz"));
    ifs_tmp.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsWarning2)
{
    std::string fn = "./support/STRU_MgO_WarningC2";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    EXPECT_NO_THROW(unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning));
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    // check warning file
    std::ifstream ifs_tmp;
    ifs_tmp.open("read_atom_positions.warn");
    std::string str((std::istreambuf_iterator<char>(ifs_tmp)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("Label read from ATOMIC_POSITIONS is Mo"));
    EXPECT_THAT(str, testing::HasSubstr("Label from ATOMIC_SPECIES is Mg"));
    ifs_tmp.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsWarning3)
{
    std::string fn = "./support/STRU_MgO_WarningC3";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_positions.tmp");
    GlobalV::ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    EXPECT_NO_THROW(unitcell::read_atom_positions(*ucell,ifa, ofs_running, GlobalV::ofs_warning));
    ofs_running.close();
    GlobalV::ofs_warning.close();
    ifa.close();
    // check warning file
    std::ifstream ifs_tmp;
    ifs_tmp.open("read_atom_positions.warn");
    std::string str((std::istreambuf_iterator<char>(ifs_tmp)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("read_atom_positions  warning :  atom number < 0."));
    ifs_tmp.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsWarning4)
{
    std::string fn = "./support/STRU_MgO_WarningC4";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    ofs_running.open("read_atom_positions.tmp");
    ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->orbital_fn.resize(ucell->ntype);
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    testing::internal::CaptureStdout();
    EXPECT_EXIT(unitcell::read_atom_positions(*ucell,ifa, ofs_running, ofs_warning), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("read_atom_positions, mismatch in atom number for atom type: Mg"));
    ofs_running.close();
    ofs_warning.close();
    ifa.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}

TEST_F(UcellTestReadStru, ReadAtomPositionsWarning5)
{
    std::string fn = "./support/STRU_MgO";
    std::ifstream ifa(fn.c_str());
    std::ofstream ofs_running;
    ofs_running.open("read_atom_positions.tmp");
    GlobalV::ofs_warning.open("read_atom_positions.warn");
    // mandatory preliminaries
    ucell->ntype = 2;
    ucell->atoms = new Atom[ucell->ntype];
    ucell->set_atom_flag = true;
    PARAM.input.test_pseudo_cell = 2;
    PARAM.input.basis_type = "lcao";
    PARAM.sys.deepks_setorb = true;
    PARAM.input.calculation = "md";
    PARAM.input.esolver_type = "arbitrary";
    EXPECT_NO_THROW(unitcell::read_atom_species(ifa, ofs_running,*ucell));
    EXPECT_NO_THROW(unitcell::read_lattice_constant(ifa, ofs_running,ucell->lat));
    EXPECT_DOUBLE_EQ(ucell->latvec.e11, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e22, 4.27957);
    EXPECT_DOUBLE_EQ(ucell->latvec.e33, 4.27957);
    // mandatory preliminaries
    delete[] ucell->magnet.start_mag;
    ucell->magnet.start_mag = new double[ucell->ntype];
    EXPECT_NO_THROW(unitcell::read_atom_positions(*ucell,ifa, ofs_running, GlobalV::ofs_warning));
    ofs_running.close();
    GlobalV::ofs_warning.close();
    ifa.close();
    // check warning file
    std::ifstream ifs_tmp;
    ifs_tmp.open("read_atom_positions.warn");
    std::string str((std::istreambuf_iterator<char>(ifs_tmp)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("read_atoms  warning : no atoms can move in MD simulations!"));
    ifs_tmp.close();
    remove("read_atom_positions.tmp");
    remove("read_atom_positions.warn");
}
#endif
