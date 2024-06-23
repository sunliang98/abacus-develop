#include "module_basis/module_nao/numerical_radial.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>

#include "module_base/constants.h"
#include "module_base/cubic_spline.h"
#include "module_base/global_variable.h"
#include "module_base/math_integral.h"
#include "module_base/spherical_bessel_transformer.h"

using ModuleBase::PI;

NumericalRadial::NumericalRadial(const NumericalRadial& other) :
    symbol_(other.symbol_),
    itype_(other.itype_),
    l_(other.l_),
    izeta_(other.izeta_),
    nr_(other.nr_),
    nk_(other.nk_),
    ircut_(other.ircut_),
    ikcut_(other.ikcut_),
    is_fft_compliant_(other.is_fft_compliant_),
    pr_(other.pr_),
    pk_(other.pk_),
    sbt_(other.sbt_)
{
    // deep copy
    if (other.rgrid())
    {
        rgrid_ = new double[nr_];
        rvalue_ = new double[nr_];
        std::memcpy(rgrid_, other.rgrid_, nr_ * sizeof(double));
        std::memcpy(rvalue_, other.rvalue_, nr_ * sizeof(double));
    }

    if (other.kgrid())
    {
        kgrid_ = new double[nk_];
        kvalue_ = new double[nk_];
        std::memcpy(kgrid_, other.kgrid_, nk_ * sizeof(double));
        std::memcpy(kvalue_, other.kvalue_, nk_ * sizeof(double));
    }
}

NumericalRadial& NumericalRadial::operator=(const NumericalRadial& rhs)
{
    if (this == &rhs)
    {
        return *this;
    }

    // wipe off r & k space data
    wipe(true, true);

    symbol_ = rhs.symbol_;
    itype_ = rhs.itype_;
    izeta_ = rhs.izeta_;
    l_ = rhs.l_;

    nr_ = rhs.nr_;
    nk_ = rhs.nk_;

    ircut_ = rhs.ircut_;
    ikcut_ = rhs.ikcut_;

    is_fft_compliant_ = rhs.is_fft_compliant_;

    pr_ = rhs.pr_;
    pk_ = rhs.pk_;

    sbt_ = rhs.sbt_;

    // deep copy
    if (rhs.rgrid())
    {
        rgrid_ = new double[nr_];
        rvalue_ = new double[nr_];
        std::memcpy(rgrid_, rhs.rgrid_, nr_ * sizeof(double));
        std::memcpy(rvalue_, rhs.rvalue_, nr_ * sizeof(double));
    }

    if (rhs.kgrid())
    {
        kgrid_ = new double[nk_];
        kvalue_ = new double[nk_];
        std::memcpy(kgrid_, rhs.kgrid_, nk_ * sizeof(double));
        std::memcpy(kvalue_, rhs.kvalue_, nk_ * sizeof(double));
    }

    return *this;
}

NumericalRadial::~NumericalRadial()
{
    delete[] rgrid_;
    delete[] kgrid_;
    delete[] rvalue_;
    delete[] kvalue_;
}

void NumericalRadial::build(const int l,
                            const bool for_r_space,
                            const int ngrid,
                            const double* const grid,
                            const double* const value,
                            const int p,
                            const int izeta,
                            const std::string symbol,
                            const int itype,
                            const bool init_sbt)
{
#ifdef __DEBUG
    assert(l >= 0);
    assert(ngrid > 1);
    assert(grid && value);

    // grid must be strictly increasing and every element must be non-negative
    assert(std::is_sorted(grid, grid + ngrid, std::less_equal<double>())); // using less_equal forbids equal values
    assert(grid[0] >= 0.0);
#endif

    // wipe off any existing r & k space data
    wipe(true, true);

    symbol_ = symbol;
    itype_ = itype;
    izeta_ = izeta;
    l_ = l;

    if (for_r_space)
    {
        nr_ = ngrid;
        pr_ = p;
        rgrid_ = new double[nr_];
        rvalue_ = new double[nr_];
        std::memcpy(rgrid_, grid, nr_ * sizeof(double));
        std::memcpy(rvalue_, value, nr_ * sizeof(double));
    }
    else
    {
        nk_ = ngrid;
        pk_ = p;
        kgrid_ = new double[nk_];
        kvalue_ = new double[nk_];
        std::memcpy(kgrid_, grid, nk_ * sizeof(double));
        std::memcpy(kvalue_, value, nk_ * sizeof(double));
    }

    set_icut(for_r_space, !for_r_space);
}

void NumericalRadial::to_numerical_orbital_lm(Numerical_Orbital_Lm& orbital_lm, const int nk_legacy, const double lcao_dk) const
{
#ifdef __DEBUG
    assert(rgrid_);
    assert(rgrid_[0] == 0.0);
    assert(is_uniform(nr_, rgrid_, 1e-14));

    // Numerical_Orbital_Lm does not support extra exponent in the real space value
    assert(pr_ == 0);
#endif

    double dr = rgrid_[1] - rgrid_[0];
    double* rab = new double[nr_];
    std::fill(rab, rab + nr_, dr);

    orbital_lm.set_orbital_info(symbol_, itype_, l_, izeta_, std::min(nr_, ircut_+1), rab, rgrid_,
            Numerical_Orbital_Lm::Psi_Type::Psi, rvalue_, nk_legacy, lcao_dk,
            0.001 /* dr_uniform */, GlobalV::out_element_info, true, GlobalV::CAL_FORCE);
    delete[] rab;
}

void NumericalRadial::set_transformer(ModuleBase::SphericalBesselTransformer sbt, int update)
{
    sbt_ = sbt;

#ifdef __DEBUG
    assert(update == 0 || update == 1 || update == -1);
#endif
    switch (update)
    {
    case 1:
        transform(true); // forward transform r -> k
        break;
    case -1:
        transform(false); // backward transform k -> r
        break;
    default:; // do nothing
    }
}

void NumericalRadial::set_grid(const bool for_r_space, const int ngrid, const double* const grid, const char mode)
{
#ifdef __DEBUG
    assert(mode == 'i' || mode == 't');
    assert(ngrid > 1);

    // grid must be strictly increasing and every element must be non-negative
    assert(std::is_sorted(grid, grid + ngrid, std::less_equal<double>())); // using less_equal forbids equal values
    assert(grid[0] >= 0.0);
#endif

    // tbu stands for "to be updated"
    double*& grid_tbu = (for_r_space ? rgrid_ : kgrid_);
    double*& value_tbu = (for_r_space ? rvalue_ : kvalue_);
    int& ngrid_tbu = (for_r_space ? nr_ : nk_);

    if (mode == 't')
    { // obtain new values by a transform from the other space
        // make sure a transform from the other space is available
#ifdef __DEBUG
        assert(for_r_space ? (kgrid_ && kvalue_) : (rgrid_ && rvalue_));
#endif

        delete[] grid_tbu;
        delete[] value_tbu;
        grid_tbu = new double[ngrid];
        value_tbu = new double[ngrid];
        ngrid_tbu = ngrid;
        std::memcpy(grid_tbu, grid, ngrid * sizeof(double));

        is_fft_compliant_ = is_fft_compliant(nr_, rgrid_, nk_, kgrid_);
        transform(!for_r_space); // transform(true): r -> k; transform(false): k -> r
        // ircut_ or ikcut_ is updated in transform()
    }
    else
    { // obtain new values by interpolation in the current space
        // make sure an interpolation in the current space is available
#ifdef __DEBUG
        assert(grid_tbu && value_tbu);
#endif

        // cubic spline interpolation
        ModuleBase::CubicSpline cubspl(ngrid_tbu, grid_tbu, value_tbu); // not-a-knot boundary condition

        double* grid_new = new double[ngrid];
        double* value_new = new double[ngrid];

        std::memcpy(grid_new, grid, ngrid * sizeof(double));
        std::fill_n(value_new, ngrid, 0.0);

        // do interpolation for grid points within the range of the origional grid
        // for grid points outside the original range, simply set the values to zero

        // grid_start is the first grid point that is greater than or equal to grid_tbu[0]
        double* grid_start = std::lower_bound(grid_new, grid_new + ngrid, grid_tbu[0]);

        // grid_end is the first grid point that is strictly greater than grid_tbu[ngrid_tbu-1]
        double* grid_end = std::upper_bound(grid_new, grid_new + ngrid, grid_tbu[ngrid_tbu - 1]);

        cubspl.eval(std::distance(grid_start, grid_end), grid_start, value_new + std::distance(grid_new, grid_start));

        delete[] grid_tbu;
        delete[] value_tbu;

        grid_tbu = grid_new;
        value_tbu = value_new;
        ngrid_tbu = ngrid;

        is_fft_compliant_ = is_fft_compliant(nr_, rgrid_, nk_, kgrid_);
        set_icut(for_r_space, !for_r_space);
        transform(for_r_space); // transform(true): r -> k; transform(false): k -> r
    }
}

void NumericalRadial::set_uniform_grid(const bool for_r_space,
                                       const int ngrid,
                                       const double cutoff,
                                       const char mode,
                                       const bool enable_fft)
{
    double* grid = new double[ngrid];
    double dx = cutoff / (ngrid - 1);
    for (int i = 0; i != ngrid; ++i)
    {
        grid[i] = i * dx;
    }

    set_grid(for_r_space, ngrid, grid, mode);
    delete[] grid;

    if (enable_fft)
    {
        set_uniform_grid(!for_r_space, ngrid, PI / dx, 't', false);
    }
}

void NumericalRadial::set_value(const bool for_r_space, const double* const value, const int p)
{
#ifdef __DEBUG
    assert(for_r_space ? rvalue_ : kvalue_);
#endif
    if (for_r_space)
    {
        std::memcpy(rvalue_, value, nr_ * sizeof(double));
        pr_ = p;
        transform(true);
        set_icut(true, false);
    }
    else
    {
        std::memcpy(kvalue_, value, nk_ * sizeof(double));
        pk_ = p;
        transform(false);
        set_icut(false, true);
    }
}

void NumericalRadial::wipe(const bool r_space, const bool k_space)
{
#ifdef __DEBUG
    assert(r_space || k_space);
#endif

    // wipe the grid and value in r/k space
    if (r_space)
    {
        delete[] rgrid_;
        delete[] rvalue_;
        rgrid_ = nullptr;
        rvalue_ = nullptr;
        nr_ = 0;
        pr_ = 0;
        ircut_ = 0;
    }

    if (k_space)
    {
        delete[] kgrid_;
        delete[] kvalue_;
        kgrid_ = nullptr;
        kvalue_ = nullptr;
        nk_ = 0;
        pk_ = 0;
        ikcut_ = 0;
    }
    is_fft_compliant_ = false;
}

void NumericalRadial::radtab(const char op,
                             const NumericalRadial& ket,
                             const int l,
                             double* const table,
                             const int nr_tab,
                             const double rmax_tab,
                             const bool deriv) const
{
#ifdef __DEBUG
    assert(op == 'S' || op == 'I' || op == 'T' || op == 'U');
    assert(l >= 0);
    assert(rmax_tab > 0 && nr_tab > 0);

    // radtab requires that two NumericalRadial objects have exactly the same (non-null) kgrid_
    assert(nk_ > 0 && nk_ == ket.nk_);
    assert(std::equal(kgrid_, kgrid_ + nk_, ket.kgrid_));
#endif

    double* rgrid_tab = new double[nr_tab];
    double dr = rmax_tab / (nr_tab - 1);
    std::for_each(rgrid_tab, rgrid_tab + nr_tab, [dr,&rgrid_tab](double& r) { r = dr * (&r - rgrid_tab); });

    bool use_radrfft = is_fft_compliant(nr_tab, rgrid_tab, nk_, kgrid_);

    // function to undergo a spherical Bessel transform:
    // overlap: chi1(k) * chi2(k)
    // kinetic: k^2 * chi1(k) * chi2(k)
    // Coulomb: k^(-2) * chi1(k) * chi2(k)
    double* fk = new double[nk_];
    std::transform(kvalue_, kvalue_ + nk_, ket.kvalue_, fk, std::multiplies<double>());

    int op_pk = 0;
    switch (op)
    {
    case 'T':
        op_pk = -2;
        break;
    case 'U':
        op_pk = 2;
        break;
    default:; // for overlap integral op_pk = 0
    }

    if (use_radrfft)
    {
        sbt_.radrfft(l, nk_, kmax(), fk, table, pk_ + ket.pk_ + op_pk);
    }
    else
    {
        sbt_.direct(l, nk_, kgrid_, fk, nr_tab, rgrid_tab, table, pk_ + ket.pk_ + op_pk);
    }

    delete[] fk;
    delete[] rgrid_tab;

    // spherical Bessel transform has a prefactor of sqrt(2/pi)
    // and the prefactor for the two-center integral radial table is 4*pi
    double pref = ModuleBase::FOUR_PI * std::sqrt(ModuleBase::PI / 2.0);
    std::for_each(table, table + nr_tab, [pref](double& x) { x *= pref; });
}

void NumericalRadial::normalize(bool for_r_space)
{
    int& ngrid = for_r_space ? nr_ : nk_;

    // tbu stands for "to be updated"
    double*& grid_tbu = for_r_space ? rgrid_ : kgrid_;
    double*& value_tbu = for_r_space ? rvalue_ : kvalue_;

    double factor = 0.0;
    double* integrand = new double[ngrid];
    double* rab = new double[ngrid];

    std::adjacent_difference(grid_tbu, grid_tbu + ngrid, rab);
    std::transform(value_tbu, value_tbu + ngrid, grid_tbu, integrand, std::multiplies<double>());
    std::for_each(integrand, integrand + ngrid, [](double& x) { x *= x; });

    factor = ModuleBase::Integral::simpson(ngrid, integrand, &rab[1]);
    factor = 1. / std::sqrt(factor);

    std::for_each(value_tbu, value_tbu + ngrid, [factor](double& x) { x *= factor; });
    transform(for_r_space);
    delete[] rab;
    delete[] integrand;
}

void NumericalRadial::transform(const bool forward)
{
#ifdef __DEBUG
    // grid & value must exist in the initial space
    assert(forward ? (rgrid_ && rvalue_) : (kgrid_ && kvalue_));
#endif

    // do nothing if there is no grid in the destination space
    if ((forward && !kgrid_) || (!forward && !rgrid_))
    {
        return;
    }

    if (forward)
    { // r -> k
        if (is_fft_compliant_)
        {
            sbt_.radrfft(l_, nr_, rgrid_[nr_ - 1], rvalue_, kvalue_, pr_);
        }
        else
        {
            sbt_.direct(l_, nr_, rgrid_, rvalue_, nk_, kgrid_, kvalue_, pr_);
        }
        pk_ = 0;
        set_icut(false, true);
    }
    else
    { // k -> r
        if (is_fft_compliant_)
        {
            sbt_.radrfft(l_, nk_, kgrid_[nk_ - 1], kvalue_, rvalue_, pk_);
        }
        else
        {
            sbt_.direct(l_, nk_, kgrid_, kvalue_, nr_, rgrid_, rvalue_, pk_);
        }
        pr_ = 0;
        set_icut(true, false);
    }
}

void NumericalRadial::set_icut(const bool for_r_space, const bool for_k_space, const double tol)
{
    if (for_r_space)
    {
#ifdef __DEBUG
        assert(rgrid_ && rvalue_);
#endif
        ircut_ = nr_;
        while (ircut_ && std::abs(rvalue_[ircut_ - 1]) <= tol) { --ircut_; }
    }

    if (for_k_space)
    {
#ifdef __DEBUG
        assert(kgrid_ && kvalue_);
#endif
        ikcut_ = nk_;
        while (ikcut_ && std::abs(kvalue_[ikcut_ - 1]) <= tol) { --ikcut_; }
    }
}

bool NumericalRadial::is_uniform(const int n, const double* const x, const double tol)
{
    double dx = (x[n - 1] - x[0]) / (n - 1);
    return std::all_of(x, x + n,
            [&](const double& xi) { return std::abs(x[0] + (&xi - x) * dx - xi) < tol; });
}

bool NumericalRadial::is_fft_compliant(const int nr,
                                       const double* const rgrid,
                                       const int nk,
                                       const double* const kgrid,
                                       const double tol
                                       )
{
    if (!rgrid || !kgrid || nr != nk || nr < 2)
    {
        return false;
    }

    double dr = rgrid[nr - 1] / (nr - 1);
    double dk = kgrid[nk - 1] / (nk - 1);

    return nr * std::abs(dr * dk - PI / (nr - 1)) < tol
           && rgrid[0] == 0.0 && is_uniform(nr, rgrid, tol)
           && kgrid[0] == 0.0 && is_uniform(nk, kgrid, tol);
}
