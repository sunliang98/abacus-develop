#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// mohan add 2025-04-12
#include "source_estate/module_charge/charge.h"

#define private public
#include "module_parameter/parameter.h"
#undef private

/************************************************
 *  unit test of magnetism.cpp
 ***********************************************/

/**
 * - Tested Functions:
 *   - Magnetism::Magnetism()
 *   - Magnetism::~Magnetism()
 *   - Magnetism::judge_parallel()
 *   - Magnetism::compute_mag()
 *      - compute mag for spin-polarized system when PARAM.input.nspin = 2
 *      - and non-collinear case with PARAM.input.nspin = 4 
*/

#define private public
#include "source_estate/magnetism.h"
#undef private
Charge::Charge()
{
}
Charge::~Charge()
{
}


class MagnetismTest : public ::testing::Test
{
  protected:
    Magnetism* magnetism;
    virtual void SetUp()
    {
        magnetism = new Magnetism;
    }
    virtual void TearDown()
    {
        delete magnetism;
    }
};

TEST_F(MagnetismTest, Magnetism)
{
    EXPECT_EQ(0.0, magnetism->tot_mag);
    EXPECT_EQ(0.0, magnetism->abs_mag);
    EXPECT_EQ(nullptr, magnetism->start_mag);
}

TEST_F(MagnetismTest, JudgeParallel)
{
    double a[3] = {1.0, 0.0, 0.0};
    ModuleBase::Vector3<double> b(1.0, 0.0, 0.0);
    EXPECT_TRUE(magnetism->judge_parallel(a, b));
    b = ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
    EXPECT_FALSE(magnetism->judge_parallel(a, b));
}

TEST_F(MagnetismTest, ComputeMagnetizationS2)
{
	PARAM.input.nspin = 2;
	PARAM.sys.two_fermi = false;
	PARAM.input.nelec = 10.0;

	Charge* chr = new Charge;
	chr->nrxx = 100;
	chr->nxyz = 1000;
	chr->rho = new double*[PARAM.input.nspin];
	for (int i=0; i< PARAM.input.nspin; i++)
	{
		chr->rho[i] = new double[chr->nrxx];
	}
	for (int ir=0; ir< chr->nrxx; ir++)
	{
		chr->rho[0][ir] = 1.00;
		chr->rho[1][ir] = 1.01;
	}
	double* nelec_spin = new double[2];
	magnetism->compute_mag(500.0,chr->nrxx, chr->nxyz, chr->rho, nelec_spin);
	EXPECT_DOUBLE_EQ(-0.5, magnetism->tot_mag);
	EXPECT_DOUBLE_EQ(0.5, magnetism->abs_mag);
	EXPECT_DOUBLE_EQ(4.75, nelec_spin[0]);
	EXPECT_DOUBLE_EQ(5.25, nelec_spin[1]);
	delete[] nelec_spin;
	for (int i=0; i< PARAM.input.nspin; i++)
	{
		delete[] chr->rho[i];
	}
	delete[] chr->rho;
	delete chr;
}

TEST_F(MagnetismTest, ComputeMagnetizationS4)
{
	PARAM.input.nspin = 4;

	Charge* chr = new Charge;
	chr->rho = new double*[PARAM.input.nspin];
	chr->nrxx = 100;
	chr->nxyz = 1000;
	for (int i=0; i< PARAM.input.nspin; i++)
	{
		chr->rho[i] = new double[chr->nrxx];
	}
	for (int ir=0; ir< chr->nrxx; ir++)
	{
		chr->rho[0][ir] = 1.00;
		chr->rho[1][ir] = std::sqrt(2.0);
		chr->rho[2][ir] = 1.00;
		chr->rho[3][ir] = 1.00;
	}
	double* nelec_spin = new double[4];
	magnetism->compute_mag(500.0,chr->nrxx, chr->nxyz, chr->rho, nelec_spin);
	EXPECT_DOUBLE_EQ(100.0, magnetism->abs_mag);
	EXPECT_DOUBLE_EQ(50.0*std::sqrt(2.0), magnetism->tot_mag_nc[0]);
	EXPECT_DOUBLE_EQ(50.0, magnetism->tot_mag_nc[1]);
	EXPECT_DOUBLE_EQ(50.0, magnetism->tot_mag_nc[2]);
	delete[] nelec_spin;
	for (int i=0; i< PARAM.input.nspin; i++)
	{
		delete[] chr->rho[i];
	}
	delete[] chr->rho;
	delete chr;
}

#ifdef __MPI
#include <mpi.h>
int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD,&GlobalV::MY_RANK);

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();

    return result;
}
#endif
