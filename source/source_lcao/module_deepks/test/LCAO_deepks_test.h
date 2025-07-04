#include "klist.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
//#include "parallel_orbitals.h"

#include "../LCAO_deepks.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>

namespace Test_Deepks
{
extern Grid_Driver GridD;
}

template <typename T>
class test_deepks
{

  public:
    test_deepks();
    ~test_deepks();

    LCAO_Orbitals ORB;

    RadialCollection orb_;
    RadialCollection alpha_;
    TwoCenterIntegrator overlap_orb_alpha_;

    UnitCell ucell;

    Parallel_Orbitals ParaO;
    Test_Deepks::K_Vectors kv;
    LCAO_Deepks<T> ld;

    int failed_check = 0;
    int total_check = 0;

    int my_rank = 0;

    double lcao_ecut = 0; // (Ry)
    double lcao_dk = 0.01;
    double lcao_dr = 0.01;
    double lcao_rmax = 30; // (a.u.)

    int out_mat_r = 0;

    int lmax = 2;
    int ntype = 0;

    using TH = std::conditional_t<std::is_same<T, double>::value, ModuleBase::matrix, ModuleBase::ComplexMatrix>;

    std::vector<TH> dm;
    std::vector<std::vector<T>> dm_new;
    elecstate::DensityMatrix<T, double>* p_elec_DM;

    // preparation
    void preparation();
    void set_parameters(); // set some global variables
    void setup_cell();

    void count_ntype(); // from STRU, count types of elements
    void set_ekcut();   // from LCAO files, read and set ekcut

    void prep_neighbour();
    void setup_kpt();
    void set_orbs();

    // tranfer Matrix into vector<T>
    void set_dm_new();

    // tranfer vector<T> into DensityMatrix
    void set_p_elec_DM();

    // checking
    void check_dstable();
    void check_phialpha();

    void read_dm(const int nks);

    void check_pdm();
    void check_descriptor(std::vector<torch::Tensor>& descriptor);

    void check_gdmx(torch::Tensor& gdmx);
    void check_gdmepsl(torch::Tensor& gdmepsl);

    void check_gvx(torch::Tensor& gdmx);
    void check_gvepsl(torch::Tensor& gdmepsl);

    void check_orbpre();

    void check_vdpre();

    void check_vdrpre();

    void check_edelta(std::vector<torch::Tensor>& descriptor);

    // calculate V_delta
    void cal_V_delta();

    void check_e_deltabands();
    void check_f_delta_and_stress_delta();
    void check_o_delta();

    // compares numbers stored in two files
    void compare_with_ref(const std::string f1, const std::string f2);
};
