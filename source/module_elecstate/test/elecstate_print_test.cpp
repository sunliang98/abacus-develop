#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#define private public
#include "module_cell/klist.h"
#include "module_elecstate/elecstate.h"
#include "module_elecstate/module_charge/charge.h"
#include "module_elecstate/module_pot/efield.h"
#include "module_elecstate/module_pot/gatefield.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_parameter/parameter.h"
#include "module_elecstate/elecstate_print.h"
#undef private 
/***************************************************************
 *  mock functions
 ****************************************************************/
namespace elecstate
{
const double* ElecState::getRho(int spin) const
{
    return &(this->eferm.ef);
} // just for mock
double Efield::etotefield = 1.1;
double elecstate::Gatefield::etotgatefield = 2.2;

} // namespace elecstate
UnitCell::UnitCell(){}
UnitCell::~UnitCell(){}
Magnetism::Magnetism(){}
Magnetism::~Magnetism(){}
InfoNonlocal::InfoNonlocal(){}
InfoNonlocal::~InfoNonlocal(){}
Charge::Charge()
{
}
Charge::~Charge()
{
}

int XC_Functional::func_type = 0;
bool XC_Functional::ked_flag = false;

/***************************************************************
 *  unit test of functions in elecstate_print.cpp
 ****************************************************************/

/**
 * - Tested functions:
 *   - ElecState::print_format()
 *   - ElecState::print_eigenvalue()
 */

class ElecStatePrintTest : public ::testing::Test
{
  protected:
    elecstate::ElecState elecstate;
    UnitCell ucell;
    std::string output;
    std::ifstream ifs;
    std::ofstream ofs;
    K_Vectors* p_klist;
    void SetUp()
    {
        p_klist = new K_Vectors;
        p_klist->set_nks(2);
        p_klist->set_nkstot(2);
        p_klist->isk = {0, 1};
        p_klist->ngk = {100, 101};
        p_klist->kvec_c.resize(2);
        p_klist->kvec_c[0].set(0.1, 0.11, 0.111);
        p_klist->kvec_c[1].set(0.2, 0.22, 0.222);
        p_klist->ik2iktot.resize(2);
        p_klist->ik2iktot[0] = 0;
        p_klist->ik2iktot[1] = 1;
        // initialize klist of elecstate
        elecstate.klist = p_klist;
        // initialize ekb of elecstate
        elecstate.ekb.create(2, 2);
        elecstate.ekb(0, 0) = 1.0;
        elecstate.ekb(0, 1) = 2.0;
        elecstate.ekb(1, 0) = 3.0;
        elecstate.ekb(1, 1) = 4.0;
        // initialize wg of elecstate
        elecstate.wg.create(2, 2);
        elecstate.wg(0, 0) = 0.1;
        elecstate.wg(0, 1) = 0.2;
        elecstate.wg(1, 0) = 0.3;
        elecstate.wg(1, 1) = 0.4;
        ucell.magnet.tot_magnetization = 1.1;
        ucell.magnet.abs_magnetization = 2.2;
        ucell.magnet.tot_magnetization_nc[0] = 3.3;
        ucell.magnet.tot_magnetization_nc[1] = 4.4;
        ucell.magnet.tot_magnetization_nc[2] = 5.5;
        PARAM.input.ks_solver = "dav";
        PARAM.sys.log_file = "test.dat";
    }
    void TearDown()
    {
        delete p_klist;
    }
};

TEST_F(ElecStatePrintTest, PrintFormat)
{
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    elecstate::print_format("test", 0.1);
    GlobalV::ofs_running.close();
    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("test                          +0.1                      +1.36057"));
    ifs.close();
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEigenvalueS2)
{
    PARAM.input.nspin = 2;
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    // print eigenvalue
    elecstate::print_eigenvalue(elecstate.ekb,elecstate.wg,elecstate.klist,GlobalV::ofs_running);
    GlobalV::ofs_running.close();
    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("STATE ENERGY(eV) AND OCCUPATIONS"));
    EXPECT_THAT(str, testing::HasSubstr("NSPIN == 2"));
    EXPECT_THAT(str, testing::HasSubstr("SPIN UP :"));
    EXPECT_THAT(str, testing::HasSubstr("1/1 kpoint (Cartesian) = 0.10000 0.11000 0.11100 (100 pws)"));
    EXPECT_THAT(str, testing::HasSubstr("1        13.6057       0.100000"));
    EXPECT_THAT(str, testing::HasSubstr("2        27.2114       0.200000"));
    EXPECT_THAT(str, testing::HasSubstr("SPIN DOWN :"));
    EXPECT_THAT(str, testing::HasSubstr("1/1 kpoint (Cartesian) = 0.20000 0.22000 0.22200 (101 pws)"));
    EXPECT_THAT(str, testing::HasSubstr("1        40.8171       0.300000"));
    EXPECT_THAT(str, testing::HasSubstr("2        54.4228       0.400000"));
    ifs.close();
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEigenvalueS4)
{
    PARAM.input.nspin = 4;
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    // print eigenvalue
    elecstate::print_eigenvalue(elecstate.ekb,elecstate.wg,elecstate.klist,GlobalV::ofs_running);
    GlobalV::ofs_running.close();
    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("STATE ENERGY(eV) AND OCCUPATIONS"));
    EXPECT_THAT(str, testing::HasSubstr("NSPIN == 4"));
    EXPECT_THAT(str, testing::HasSubstr("1/2 kpoint (Cartesian) = 0.10000 0.11000 0.11100 (100 pws)"));
    EXPECT_THAT(str, testing::HasSubstr("1        13.6057       0.100000"));
    EXPECT_THAT(str, testing::HasSubstr("2        27.2114       0.200000"));
    EXPECT_THAT(str, testing::HasSubstr("2/2 kpoint (Cartesian) = 0.20000 0.22000 0.22200 (101 pws)"));
    EXPECT_THAT(str, testing::HasSubstr("1        40.8171       0.300000"));
    EXPECT_THAT(str, testing::HasSubstr("2        54.4228       0.400000"));
    ifs.close();
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintBand)
{
    PARAM.input.nspin = 1;
    PARAM.input.nbands = 2;
    PARAM.sys.nbands_l = 2;
    GlobalV::MY_RANK = 0;

    std::ofstream ofs;
    ofs.open("test.dat", std::ios::out);
    // print eigenvalues
    elecstate::print_band(elecstate.ekb,elecstate.wg,elecstate.klist, 0, 1, 0, ofs);
    ofs.close();

    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("Energy (eV) & Occupations for spin=1 k-point=1"));
    EXPECT_THAT(str, testing::HasSubstr("1        13.6057       0.100000"));
    EXPECT_THAT(str, testing::HasSubstr("2        27.2114       0.200000"));
    ifs.close();
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEigenvalueWarning)
{
    elecstate.ekb(0, 0) = 1.0e11;
    PARAM.input.nspin = 4;
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    testing::internal::CaptureStdout();
    EXPECT_EXIT(elecstate::print_eigenvalue(elecstate.ekb,elecstate.wg,elecstate.klist,GlobalV::ofs_running), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("Eigenvalues are too large!"));
    GlobalV::ofs_running.close();
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintBandWarning)
{
    elecstate.ekb(0, 0) = 1.0e11;
    PARAM.input.nspin = 4;

    std::ofstream ofs;
    ofs.open("test.dat", std::ios::out);
    testing::internal::CaptureStdout();
    
    EXPECT_EXIT(elecstate::print_band(elecstate.ekb,elecstate.wg,elecstate.klist, 0, 1, 0, ofs), 
      ::testing::ExitedWithCode(1), "");

    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("Eigenvalues are too large!"));

    ofs.close();

    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEtot)
{
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    bool converged = false;
    int iter = 1;
    double scf_thr = 0.1;
    double scf_thr_kin = 0.0;
    double duration = 2.0;
    int printe = 1;
    double pw_diag_thr = 0.1;
    int avg_iter = 2;
    bool print = true;
    elecstate.charge = new Charge;
    elecstate.charge->nrxx = 100;
    elecstate.charge->nxyz = 1000;
    PARAM.input.imp_sol = true;
    PARAM.input.efield_flag = true;
    PARAM.input.gate_flag = true;
    PARAM.sys.two_fermi = true;
    PARAM.input.out_bandgap = true;
    GlobalV::MY_RANK = 0;
    PARAM.input.basis_type = "pw";
    PARAM.input.nspin = 2;
    // iteration of different vdw_method
    std::vector<std::string> vdw_methods = {"d2", "d3_0", "d3_bj"};
    for (int i = 0; i < vdw_methods.size(); i++)
    {
        PARAM.input.vdw_method = vdw_methods[i];
        elecstate::print_etot(ucell.magnet,elecstate, converged, iter, scf_thr, scf_thr_kin, duration, printe, pw_diag_thr, avg_iter, false);
    }
    // iteration of different ks_solver
    std::vector<std::string> ks_solvers = {"cg", "lapack", "genelpa", "dav", "scalapack_gvx", "cusolver"};
    for (int i = 0; i < ks_solvers.size(); i++)
    {
        PARAM.input.ks_solver = ks_solvers[i];
        testing::internal::CaptureStdout();
        elecstate::print_etot(ucell.magnet,elecstate,converged, iter, scf_thr, scf_thr_kin, duration, printe, pw_diag_thr, avg_iter, print);
        output = testing::internal::GetCapturedStdout();
        if (PARAM.input.ks_solver == "cg")
        {
            EXPECT_THAT(output, testing::HasSubstr("CG"));
        }
        else if (PARAM.input.ks_solver == "lapack")
        {
            EXPECT_THAT(output, testing::HasSubstr("LA"));
        }
        else if (PARAM.input.ks_solver == "genelpa")
        {
            EXPECT_THAT(output, testing::HasSubstr("GE"));
        }
        else if (PARAM.input.ks_solver == "dav")
        {
            EXPECT_THAT(output, testing::HasSubstr("DA"));
        }
        else if (PARAM.input.ks_solver == "scalapack_gvx")
        {
            EXPECT_THAT(output, testing::HasSubstr("GV"));
        }
        else if (PARAM.input.ks_solver == "cusolver")
        {
            EXPECT_THAT(output, testing::HasSubstr("CU"));
        }
    }
    GlobalV::ofs_running.close();
    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("Electron density error is 0.1"));
    EXPECT_THAT(str, testing::HasSubstr("Error Threshold = 0.1"));
    EXPECT_THAT(str, testing::HasSubstr("E_KohnSham"));
    EXPECT_THAT(str, testing::HasSubstr("E_vdwD2"));
    EXPECT_THAT(str, testing::HasSubstr("E_vdwD3"));
    EXPECT_THAT(str, testing::HasSubstr("E_sol_el"));
    EXPECT_THAT(str, testing::HasSubstr("E_sol_cav"));
    EXPECT_THAT(str, testing::HasSubstr("E_efield"));
    EXPECT_THAT(str, testing::HasSubstr("E_gatefield"));
    ifs.close();
    delete elecstate.charge;
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEtot2)
{
    GlobalV::ofs_running.open("test.dat", std::ios::out);
    bool converged = false;
    int iter = 1;
    double scf_thr = 0.1;
    double scf_thr_kin = 0.0;
    double duration = 2.0;
    int printe = 0;
    double pw_diag_thr = 0.1;
    int avg_iter = 2;
    bool print = true;
    elecstate.charge = new Charge;
    elecstate.charge->nrxx = 100;
    elecstate.charge->nxyz = 1000;
    PARAM.input.imp_sol = true;
    PARAM.input.efield_flag = true;
    PARAM.input.gate_flag = true;
    PARAM.sys.two_fermi = false;
    PARAM.input.out_bandgap = true;
    GlobalV::MY_RANK = 0;
    PARAM.input.basis_type = "pw";
    PARAM.input.scf_nmax = 100;

    elecstate::print_etot(ucell.magnet,elecstate,converged, iter, scf_thr, scf_thr_kin, duration, printe, pw_diag_thr, avg_iter, print);
    GlobalV::ofs_running.close();
    ifs.open("test.dat", std::ios::in);
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("Electron density error is 0.1"));
    EXPECT_THAT(str, testing::HasSubstr("Error Threshold = 0.1"));
    EXPECT_THAT(str, testing::HasSubstr("E_KohnSham"));
    EXPECT_THAT(str, testing::HasSubstr("E_Harris"));
    EXPECT_THAT(str, testing::HasSubstr("E_Fermi"));
    EXPECT_THAT(str, testing::HasSubstr("E_bandgap"));
    ifs.close();
    delete elecstate.charge;
    std::remove("test.dat");
}

TEST_F(ElecStatePrintTest, PrintEtotColorS2)
{
    bool converged = false;
    int iter = 1;
    double scf_thr = 2.0;
    double scf_thr_kin = 0.0;
    double duration = 2.0;
    int printe = 1;
    double pw_diag_thr = 0.1;
    int avg_iter = 2;
    bool print = true;
    elecstate.charge = new Charge;
    elecstate.charge->nrxx = 100;
    elecstate.charge->nxyz = 1000;
    PARAM.input.imp_sol = true;
    PARAM.input.efield_flag = true;
    PARAM.input.gate_flag = true;
    PARAM.sys.two_fermi = true;
    PARAM.input.out_bandgap = true;
    PARAM.input.nspin = 2;
    GlobalV::MY_RANK = 0;
    elecstate::print_etot(ucell.magnet,elecstate,converged, iter, scf_thr, scf_thr_kin, duration, printe, pw_diag_thr, avg_iter, print);
    delete elecstate.charge;
}

TEST_F(ElecStatePrintTest, PrintEtotColorS4)
{
    bool converged = false;
    int iter = 1;
    double scf_thr = 0.1;
    double scf_thr_kin = 0.0;
    double duration = 2.0;
    int printe = 1;
    double pw_diag_thr = 0.1;
    int avg_iter = 2;
    bool print = true;
    elecstate.charge = new Charge;
    elecstate.charge->nrxx = 100;
    elecstate.charge->nxyz = 1000;
    PARAM.input.imp_sol = true;
    PARAM.input.efield_flag = true;
    PARAM.input.gate_flag = true;
    PARAM.sys.two_fermi = true;
    PARAM.input.out_bandgap = true;
    PARAM.input.nspin = 4;
    PARAM.input.noncolin = true;
    GlobalV::MY_RANK = 0;
    elecstate::print_etot(ucell.magnet,elecstate, converged, iter, scf_thr, scf_thr_kin, duration, printe, pw_diag_thr, avg_iter, print);
    delete elecstate.charge;
}
