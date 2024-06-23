#include <cmath>
#include <memory>
#include <fftw3.h>

#include "gtest/gtest.h"
#include "module_base/spherical_bessel_transformer.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include "module_base/constants.h"
#include "module_basis/module_nao/numerical_radial.h"

using ModuleBase::PI;
using ModuleBase::SphericalBesselTransformer;

/***********************************************************
 *      Unit test of class "NumericalRadial"
 ***********************************************************/
/*!
 *  Tested functions:
 *
 *  - build
 *      - Initializes the object by setting the grid & values in one space.
 *
 *  - to_numerical_orbital_lm
 *      - Overwrites the content of a Numerical_Orbital_Lm object with the current object.
 *
 *  - set_transformer
 *      - Sets a SphericalBesselTransformer for the object.
 *
 *  - set_grid
 *      - Sets up a new grid.
 *
 *  - set_uniform_grid
 *      - Sets up a new uniform grid.
 *
 *  - set_value
 *      - Updates values on an existing grid.
 *
 *  - wipe
 *      - Removes the grid & values from one space.
 *
 *  - radtab
 *      - Computes the radial table for two-center integrals
 *        between "this" and another object.
 *
 *  - all "getters"
 *      - Get access to private members.
 *                                                          */
class NumericalRadialTest : public ::testing::Test
{
  protected:
    void SetUp();
    void TearDown();

    int sz_max = 10000;     //!< size of each buffer
    double* grid = nullptr; //!< buffer for input grid
    double* f = nullptr;    //!< buffer for input values
    double* g = nullptr;    //!< buffer for reference values

    NumericalRadial chi; //!< object under test

    double tol = 1e-8; //!< tolerance for element-wise numerical error
};

void NumericalRadialTest::SetUp()
{
    grid = new double[sz_max];
    f = new double[sz_max];
    g = new double[sz_max];
}

void NumericalRadialTest::TearDown()
{
    delete[] f;
    delete[] g;
    delete[] grid;
}

TEST_F(NumericalRadialTest, ConstructAndAssign)
{
    /*
     * Tests the copy constructor and copy assignment operator.
     *                                                                      */
    double dk = PI / 50;
    int sz = 10000;
    int pk = -2;
    double pref = 48 * std::sqrt(2. / PI);
    for (int ik = 0; ik != sz; ++ik)
    {
        double k = ik * dk;
        grid[ik] = k;
        f[ik] = pref / std::pow(k * k + 1, 4);
    }

    chi.build(2, false, sz, grid, f, pk);
    chi.set_uniform_grid(true, sz, PI / dk, 't');

    NumericalRadial chi2(chi);
    EXPECT_EQ(chi.symbol(), chi2.symbol());
    EXPECT_EQ(chi.izeta(), chi2.izeta());
    EXPECT_EQ(chi.itype(), chi2.itype());
    EXPECT_EQ(chi.l(), chi2.l());

    EXPECT_EQ(chi.nr(), chi2.nr());
    EXPECT_EQ(chi.nk(), chi2.nk());
    EXPECT_EQ(chi.rcut(), chi2.rcut());
    EXPECT_EQ(chi.kcut(), chi2.kcut());

    ASSERT_NE(chi2.rgrid(), nullptr);
    ASSERT_NE(chi2.rvalue(), nullptr);
    for (int ir = 0; ir != sz; ++ir)
    {
        EXPECT_EQ(chi.rgrid(ir), chi2.rgrid(ir));
        EXPECT_EQ(chi.rvalue(ir), chi2.rvalue(ir));
    }

    ASSERT_NE(chi2.kgrid(), nullptr);
    ASSERT_NE(chi2.kvalue(), nullptr);
    for (int ik = 0; ik != sz; ++ik)
    {
        EXPECT_EQ(chi.kgrid(ik), chi2.kgrid(ik));
        EXPECT_EQ(chi.kvalue(ik), chi2.kvalue(ik));
    }

    EXPECT_EQ(chi.pr(), chi2.pr());
    EXPECT_EQ(chi.pk(), chi2.pk());
    EXPECT_EQ(chi.is_fft_compliant(), chi2.is_fft_compliant());
    EXPECT_EQ(chi2.sbt(), chi.sbt());

    NumericalRadial chi3;
    chi3 = chi;
    EXPECT_EQ(chi.symbol(), chi3.symbol());
    EXPECT_EQ(chi.izeta(), chi3.izeta());
    EXPECT_EQ(chi.itype(), chi3.itype());
    EXPECT_EQ(chi.l(), chi3.l());

    EXPECT_EQ(chi.nr(), chi3.nr());
    EXPECT_EQ(chi.nk(), chi3.nk());
    EXPECT_EQ(chi.rcut(), chi3.rcut());
    EXPECT_EQ(chi.kcut(), chi3.kcut());

    ASSERT_NE(chi3.rgrid(), nullptr);
    ASSERT_NE(chi3.rvalue(), nullptr);
    for (int ir = 0; ir != sz; ++ir)
    {
        EXPECT_EQ(chi.rgrid(ir), chi3.rgrid(ir));
        EXPECT_EQ(chi.rvalue(ir), chi3.rvalue(ir));
    }

    ASSERT_NE(chi3.kgrid(), nullptr);
    ASSERT_NE(chi3.kvalue(), nullptr);
    for (int ik = 0; ik != sz; ++ik)
    {
        EXPECT_EQ(chi.kgrid(ik), chi3.kgrid(ik));
        EXPECT_EQ(chi.kvalue(ik), chi3.kvalue(ik));
    }

    EXPECT_EQ(chi.pr(), chi3.pr());
    EXPECT_EQ(chi.pk(), chi3.pk());
    EXPECT_EQ(chi.is_fft_compliant(), chi3.is_fft_compliant());

    SphericalBesselTransformer sbt;
    chi.set_transformer(sbt, 1);
    chi3 = chi;
    EXPECT_EQ(chi3.sbt(), chi.sbt());

    // self assignment is not common, but it should not throw
    EXPECT_NO_THROW(chi3 = chi3);
}

TEST_F(NumericalRadialTest, BuildAndGet)
{
    /*
     * Builds a NumericalRadial object and gets access to its members.
     *                                                                      */
    int l = 1;
    double dr = 0.01;
    int sz = 5000;
    int pr = -1;
    int itype = 3;
    int izeta = 5;
    std::string symbol = "Au";
    for (int ir = 0; ir != sz; ++ir)
    {
        double r = ir * dr;
        grid[ir] = r;
        f[ir] = std::exp(-r);
    }

    chi.build(l, true, sz, grid, f, pr, izeta, symbol, itype);

    EXPECT_EQ(chi.symbol(), symbol);
    EXPECT_EQ(chi.izeta(), izeta);
    EXPECT_EQ(chi.itype(), itype);
    EXPECT_EQ(chi.l(), l);

    EXPECT_EQ(chi.nr(), sz);
    EXPECT_EQ(chi.nk(), 0);
    EXPECT_EQ(chi.rmax(), grid[sz - 1]);

    ASSERT_NE(chi.rgrid(), nullptr);
    ASSERT_NE(chi.rvalue(), nullptr);
    for (int ir = 0; ir != sz; ++ir)
    {
        EXPECT_EQ(chi.rgrid(ir), grid[ir]);
        EXPECT_EQ(chi.rvalue(ir), f[ir]);
    }

    EXPECT_EQ(chi.kgrid(), nullptr);
    EXPECT_EQ(chi.kvalue(), nullptr);

    EXPECT_EQ(chi.pr(), pr);
    EXPECT_EQ(chi.pk(), 0);
    EXPECT_EQ(chi.is_fft_compliant(), false);
}

TEST_F(NumericalRadialTest, GridSetAndWipe)
{
    /*
     * This test first builds a NumericalRadial object with r-space values
     *
     *                  r*exp(-r)
     *
     * Next, a k-space grid is set up and the k-space values are compared
     * with the analytic expression
     *
     *          sqrt(2/pi) * 8 * k / (k^2+1)^3.
     *
     * Finally, the grid & values are wiped off in both r & k space.
     *
     * NOTE: r & k grids in this test are not FFT-compliant.
     *                                                                      */
    double dr = 0.01;
    int nr = 5000;
    int pr = -1;
    for (int ir = 0; ir != nr ; ++ir)
    {
        double r = ir * dr;
        grid[ir] = r;
        f[ir] = std::exp(-r);
    }

    chi.build(1, true, nr, grid, f, pr);

    int nk = 2000;
    double* kgrid = new double[nk];
    double dk = 0.01;

    for (int ik = 0; ik != nk; ++ik)
    {
        kgrid[ik] = ik * dk;
    }

    chi.set_grid(false, nk, kgrid, 't');

    double pref = 8 * std::sqrt(2. / PI);
    for (int ik = 0; ik != nk; ++ik)
    {
        double k = ik * dk;
        EXPECT_NEAR(pref * k / std::pow(k * k + 1, 3), chi.kvalue(ik), tol);
    }

    EXPECT_EQ(chi.is_fft_compliant(), false);

    chi.wipe(true);
    EXPECT_EQ(chi.rgrid(), nullptr);
    EXPECT_EQ(chi.rvalue(), nullptr);
    EXPECT_EQ(chi.nr(), 0);
    EXPECT_EQ(chi.is_fft_compliant(), false);

    chi.wipe(false);
    EXPECT_EQ(chi.kgrid(), nullptr);
    EXPECT_EQ(chi.kvalue(), nullptr);
    EXPECT_EQ(chi.nk(), 0);

    delete[] kgrid;
}

TEST_F(NumericalRadialTest, SetUniformGrid)
{
    /*
     * This test starts from a NumericalRadial object with k-space values of
     *
     *          48*sqrt(2/pi) * k^2  / (k^2+1)^4.
     *
     * A uniform r-space grid is then set up, and values on the grid is checked
     * with the analytic expression
     *
     *          r^2 * exp(-r)
     *                                                                      */
    double dk = PI / 50;
    int sz = 10000;
    int pk = -2;
    double pref = 48 * std::sqrt(2. / PI);
    for (int ik = 0; ik != sz; ++ik)
    {
        double k = ik * dk;
        grid[ik] = k;
        f[ik] = pref / std::pow(k * k + 1, 4);
    }

    chi.build(2, false, sz, grid, f, pk);
    chi.set_uniform_grid(true, sz, PI / dk, 't', true);

    double dr = PI / chi.kmax();
    for (int ir = 0; ir != sz; ++ir)
    {
        double r = ir * dr;
        EXPECT_NEAR(r * r * std::exp(-r), chi.rvalue(ir), tol);
    }
}

TEST_F(NumericalRadialTest, Interpolate) {
    /*
     * This test starts with a NumericalRadial object with k-space values
     *
     *          48*sqrt(2/pi) * k^2  / (k^2+1)^4
     *
     * on a non-uniform k-grid. A uniform k-grid is then set up with values
     * obtained by interpolation. Finally, a FFT-compliant r-grid is set up,
     * and the r-space values are checked with the analytic expression
     *
     *          r^2 * exp(-r)
     *                                                                      */
    double dk = 0.01;
    int sz = 10000;
    int pk = -2;
    double pref = 48 * std::sqrt(2./PI);
    for (int ik = 0; ik != sz; ++ik) {
        double k = ik * dk;
        k *= std::exp(0.02*k);
        grid[ik] = k;
        f[ik] = pref / std::pow(k*k+1, 4);
    }

    chi.build(2, false, sz, grid, f, pk);

    chi.set_uniform_grid(false, sz, PI/50*(sz-1), 'i', true);

    double dr = PI / chi.kmax();
    for (int ir = 0; ir != sz; ++ir)
    {
        double r = ir * dr;
        EXPECT_NEAR(r*r*std::exp(-r), chi.rvalue(ir), tol*2); // slightly relax the tolerance due to interpolation
    }
}

TEST_F(NumericalRadialTest, ZeroPadding) {
    /*
     * This test checks whether set_grid properly pads the value array.
     *                                                                      */
    double dk = PI / 50;
    int sz1 = 2000;
    int pk = -2;
    double pref = 48 * std::sqrt(2. / PI);
    for (int ik = 0; ik != sz1; ++ik)
    {
        double k = ik * dk;
        grid[ik] = k;
        f[ik] = pref / std::pow(k * k + 1, 4);
    }

    chi.build(2, false, sz1, grid, f, pk);

    int sz2 = 10000;
    chi.set_uniform_grid(false, sz2, dk*(sz2-1), 'i');

    for (int ik = 0; ik != sz1; ++ik)
    {
        EXPECT_EQ(f[ik], chi.kvalue(ik));
    }

    for (int ik = sz1; ik != sz2; ++ik)
    {
        EXPECT_EQ(0.0, chi.kvalue(ik));
    }
}

TEST_F(NumericalRadialTest, SetValue)
{
    /*
     * This test attempts to updates values in a NumericalRadial object.
     *                                                                      */
    double dx = 0.01;
    int sz = 5000;
    int p = -1;
    for (int i = 0; i != sz; ++i)
    {
        double r = i * dx;
        grid[i] = r;
        f[i] = std::exp(-r);
    }

    int sz_cut = 20;
    std::fill(f + sz_cut, f + sz, 0.0);

    chi.build(1, true, sz, grid, f, p);

    EXPECT_EQ(chi.rcut(), sz_cut * dx);
    EXPECT_EQ(chi.rmax(), (sz-1) * dx);

    for (int ir = 0; ir != sz; ++ir)
    {
        f[ir] *= 2;
    }
    chi.set_value(true, f, p);

    for (int i = 0; i != sz; ++i)
    {
        EXPECT_EQ(chi.rvalue(i), f[i]);
    }

    chi.build(1, false, sz, grid, f, p);
    for (int i = 0; i != sz; ++i)
    {
        f[i] *= 2;
    }
    chi.set_value(false, f, p);

    for (int i = 0; i != sz; ++i)
    {
        EXPECT_EQ(chi.kvalue(i), f[i]);
    }
}

TEST_F(NumericalRadialTest, RadialTable)
{
    /*
     * This test checks the radial table for the two-center integral
     * between the following two radial functions:
     *
     *      chi1(r) = exp(-r^2)                         l = 0
     *      chi1(k) = sqrt(2)/4 * exp(-k^2/4)
     *
     *      chi2(r) = r^2 * exp(-r^2)                   l = 2
     *      chi2(k) = sqrt(2)/16 * k^2 * exp(-k^2/4)
     *
     * and compares the results with the analytic expressions:
     * (below c = 4*pi*sqrt(pi/2))
     *
     *      S(l=0, R) = c * (3-R^2)/32 * exp(-R^2/2)
     *      S(l=2, R) = c * R^2/32 * exp(-R^2/2)
     *      T(l=0, R) = c * (x^4-10*x^2+15)/32 * exp(-R^2/2)
     *      U(l=0, R) = c * 1/32 * exp(-R*R/2)
     *
     *                                                                      */
    double pref = std::sqrt(2) / 16;
    int sz = 5000;
    double dr = 0.01;
    double dk = PI / ((sz - 1) * dr);
    for (int ir = 0; ir != sz; ++ir)
    {
        double r = ir * dr;
        grid[ir] = r;
        f[ir] = std::exp(-r * r);
    }

    NumericalRadial chi1, chi2;
    chi1.build(0, true, sz, grid, f, 0);
    chi2.build(2, true, sz, grid, f, -2);

    chi1.set_uniform_grid(false, sz, PI / dr, 't');
    chi2.set_uniform_grid(false, sz, PI / dr, 't');

    // make sure chi(k) have expected values
    for (int ik = 1; ik != sz; ++ik)
    {
        double k = ik * dk;
        ASSERT_NEAR(chi1.kvalue(ik), 4 * pref * std::exp(-k * k / 4), tol);
        ASSERT_NEAR(chi2.kvalue(ik), pref * k * k * std::exp(-k * k / 4), tol);
    }

    double* table = new double[sz];
    double table_pref = ModuleBase::FOUR_PI * std::sqrt(ModuleBase::PI / 2.0);
    double rmax_tab = chi1.rmax();

    chi1.radtab('S', chi2, 0, table, chi1.nr(), rmax_tab);
    for (int i = 0; i != sz; ++i)
    {
        double R = i * dr;
        EXPECT_NEAR(table[i], table_pref * (3 - R * R) / 32 * std::exp(-R * R / 2), tol);
    }

    chi1.radtab('S', chi2, 2, table, chi1.nr(), rmax_tab);
    for (int i = 0; i != sz; ++i)
    {
        double R = i * dr;
        EXPECT_NEAR(table[i], table_pref * R * R / 32 * std::exp(-R * R / 2), tol);
    }

    chi1.radtab('T', chi2, 0, table, chi1.nr(), rmax_tab);
    for (int i = 0; i != sz; ++i)
    {
        double R = i * dr;
        EXPECT_NEAR(table[i], table_pref * (std::pow(R, 4) - 10 * R * R + 15) / 32 * std::exp(-R * R / 2), tol);
    }

    chi1.radtab('U', chi2, 0, table, chi1.nr(), rmax_tab);
    for (int i = 0; i != sz; ++i)
    {
        double R = i * dr;
        EXPECT_NEAR(table[i], table_pref * 1. / 32 * std::exp(-R * R / 2), tol);
    }

    delete[] table;
}

TEST_F(NumericalRadialTest, ToNumericalOrbitalLm)
{
    /*
     * Builds a Numerical_Orbital_Lm object from a NumericalRadial object.
     *                                                                      */
    int l = 1;
    double dr = 0.01;
    int nr = 5000;
    int pr = 0;
    int itype = 3;
    int izeta = 5;
    std::string symbol = "Au";
    for (int ir = 0; ir != nr; ++ir)
    {
        double r = ir * dr;
        grid[ir] = r;
        f[ir] = std::exp(-r);
    }

    chi.build(l, true, nr, grid, f, pr, izeta, symbol, itype);

    int nk = 1001;
    double kcut = 30;
    chi.set_uniform_grid(false, nk, kcut, 't');

    Numerical_Orbital_Lm nol;
    double lcao_ecut = 100;
    double lcao_dk = 0.01;

    int nk_legacy = static_cast<int>(std::sqrt(lcao_ecut) / lcao_dk) + 4;
    nk_legacy += 1 - nk_legacy % 2;

    double kcut_legacy = (nk_legacy - 1) * lcao_dk;

    chi.to_numerical_orbital_lm(nol, nk_legacy, lcao_dk);
    int nrcut = static_cast<int>(chi.rcut() / dr) + 1;

    // check that the orbital_lm has the same values as the chi
    EXPECT_EQ(nol.getLabel(), symbol);
    EXPECT_EQ(nol.getType(), itype);
    EXPECT_EQ(nol.getL(), l);
    EXPECT_EQ(nol.getChi(), izeta);
    EXPECT_EQ(nol.getNr(), nrcut);
    EXPECT_EQ(nol.getNk(), nk_legacy);

    EXPECT_EQ(nol.getRcut(), chi.rcut());
    EXPECT_EQ(nol.getKcut(), kcut_legacy);

    EXPECT_EQ(nol.getRadial(111), grid[111]);
    EXPECT_EQ(nol.getRadial(777), grid[777]);
    EXPECT_EQ(nol.getKpoint(3), 3 * lcao_dk);

    EXPECT_EQ(nol.getRab(123), dr);
    EXPECT_EQ(nol.getDk(), lcao_dk);

    EXPECT_EQ(nol.getPsi(55), f[55]);
    EXPECT_EQ(nol.getPsi(222), f[222]);
    EXPECT_EQ(nol.getPsi(3333), f[3333]);
    // k values may have noticable difference due to algorithmic distinction
}

int main(int argc, char** argv)
{

#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    fftw_cleanup();

    return result;
}
