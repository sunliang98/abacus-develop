#include "math_erf_complex.h"

#include "constants.h"

#include <cfloat>
#include <limits>
#define Inf std::numeric_limits<double>::infinity()
#define NaN std::numeric_limits<double>::quiet_NaN()

namespace ModuleBase
{

ErrorFunc::ErrorFunc()
{
}
ErrorFunc::~ErrorFunc()
{
}

std::complex<double> ErrorFunc::scaled_w(std::complex<double> z, double relerr)
{
    if (std::real(z) == 0.0)
        return std::complex<double>(erfcx(std::imag(z)), std::real(z));
    else if (std::imag(z) == 0.0)
        return std::complex<double>(std::exp(-std::real(z) * std::real(z)), scaled_w_im(std::real(z)));

    double a, a2, c;
    if (relerr <= DBL_EPSILON)
    {
        relerr = DBL_EPSILON;
        a = 0.518321480430085929872;  // pi / sqrt(-log(eps*0.5))
        c = 0.329973702884629072537;  // (2/pi) * a;
        a2 = 0.268657157075235951582; // a^2
    }
    else
    {
        if (relerr > 0.1)
            relerr = 0.1; // not sensible to compute < 1 digit
        a = ModuleBase::PI / std::sqrt(-std::log(relerr * 0.5));
        c = (2 / ModuleBase::PI) * a;
        a2 = a * a;
    }
    const double x = std::fabs(std::real(z));
    const double y = std::imag(z), ya = std::fabs(y);
    std::complex<double> ret = 0.; // return value
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    if (ya > 7 || (x > 6 && (ya > 0.1 || (x > 8 && ya > 1e-10) || x > 28)))
    {

        const double ispi = 1 / std::sqrt(ModuleBase::PI); // 1 / sqrt(pi)
        double xs = y < 0 ? -std::real(z) : std::real(z);  // compute for -z if y < 0
        if (x + ya > 4000)
        { // nu <= 2
            if (x + ya > 1e7)
            { // nu == 1, w(z) = i/sqrt(pi) / z
                // scale to avoid overflow
                if (x > ya)
                {
                    double yax = ya / xs;
                    double denom = ispi / (xs + yax * ya);
                    ret = std::complex<double>(denom * yax, denom);
                }
                else if (std::isinf(ya))
                    return ((std::isnan(x) || y < 0) ? std::complex<double>(NaN, NaN) : std::complex<double>(0, 0));
                else
                {
                    double xya = xs / ya;
                    double denom = ispi / (xya * xs + ya);
                    ret = std::complex<double>(denom, denom * xya);
                }
            }
            else
            { // nu == 2, w(z) = i/sqrt(pi) * z / (z*z - 0.5)
                double dr = xs * xs - ya * ya - 0.5, di = 2 * xs * ya;
                double denom = ispi / (dr * dr + di * di);
                ret = std::complex<double>(denom * (xs * di - ya * dr), denom * (xs * dr + ya * di));
            }
        }
        else
        { // compute nu(z) estimate and do general continued fraction
            const double c0 = 3.9, c1 = 11.398, c2 = 0.08254, c3 = 0.1421, c4 = 0.2023; // fit
            double nu = std::floor(c0 + c1 / (c2 * x + c3 * ya + c4));
            double wr = xs, wi = ya;
            for (nu = 0.5 * (nu - 1); nu > 0.4; nu -= 0.5)
            {
                // w <- z - nu/w:
                double denom = nu / (wr * wr + wi * wi);
                wr = xs - wr * denom;
                wi = ya + wi * denom;
            }
            { // w(z) = i/sqrt(pi) / w:
                double denom = ispi / (wr * wr + wi * wi);
                ret = std::complex<double>(denom * wi, denom * wr);
            }
        }
        if (y < 0)
        {
            // use w(z) = 2.0*exp(-z*z) - w(-z),
            // but be careful of overflow in exp(-z*z)
            //                                = exp(-(xs*xs-ya*ya) -2*i*xs*ya)
            return 2.0 * std::exp(std::complex<double>((ya - xs) * (xs + ya), 2 * xs * y)) - ret;
        }
        else
            return ret;
    }
    else if (x < 10)
    {
        double prod2ax = 1, prodm2ax = 1;
        double expx2;

        if (std::isnan(y))
            return std::complex<double>(y, y);

        if (relerr == DBL_EPSILON)
        { // use precomputed exp(-a2*(n*n)) table
            if (x < 5e-4)
            { // compute sum4 and sum5 together as sum5-sum4
                const double x2 = x * x;
                expx2 = 1 - x2 * (1 - 0.5 * x2); // exp(-x*x) via Taylor
                // compute exp(2*a*x) and exp(-2*a*x) via Taylor, to double precision
                const double ax2 = 1.036642960860171859744 * x; // 2*a*x
                const double exp2ax = 1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667 * ax2));
                const double expm2ax = 1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667 * ax2));
                for (int n = 1; 1; ++n)
                {
                    const double coef = expa2n2[n - 1] * expx2 / (a2 * (n * n) + y * y);
                    prod2ax *= exp2ax;
                    prodm2ax *= expm2ax;
                    sum1 += coef;
                    sum2 += coef * prodm2ax;
                    sum3 += coef * prod2ax;

                    // really = sum5 - sum4
                    sum5 += coef * (2 * a) * n * std::sinh((2 * a) * n * x);

                    // test convergence via sum3
                    if (coef * prod2ax < relerr * sum3)
                        break;
                }
            }
            else
            { // x > 5e-4, compute sum4 and sum5 separately
                expx2 = std::exp(-x * x);
                const double exp2ax = std::exp((2 * a) * x), expm2ax = 1 / exp2ax;
                for (int n = 1; 1; ++n)
                {
                    const double coef = expa2n2[n - 1] * expx2 / (a2 * (n * n) + y * y);
                    prod2ax *= exp2ax;
                    prodm2ax *= expm2ax;
                    sum1 += coef;
                    sum2 += coef * prodm2ax;
                    sum4 += (coef * prodm2ax) * (a * n);
                    sum3 += coef * prod2ax;
                    sum5 += (coef * prod2ax) * (a * n);
                    // test convergence via sum5, since this sum has the slowest decay
                    if ((coef * prod2ax) * (a * n) < relerr * sum5)
                        break;
                }
            }
        }
        else
        { // relerr != DBL_EPSILON, compute exp(-a2*(n*n)) on the fly
            const double exp2ax = std::exp((2 * a) * x), expm2ax = 1 / exp2ax;
            if (x < 5e-4)
            { // compute sum4 and sum5 together as sum5-sum4
                const double x2 = x * x;
                expx2 = 1 - x2 * (1 - 0.5 * x2); // exp(-x*x) via Taylor
                for (int n = 1; 1; ++n)
                {
                    const double coef = exp(-a2 * (n * n)) * expx2 / (a2 * (n * n) + y * y);
                    prod2ax *= exp2ax;
                    prodm2ax *= expm2ax;
                    sum1 += coef;
                    sum2 += coef * prodm2ax;
                    sum3 += coef * prod2ax;

                    // really = sum5 - sum4
                    sum5 += coef * (2 * a) * n * std::sinh((2 * a) * n * x);

                    // test convergence via sum3
                    if (coef * prod2ax < relerr * sum3)
                        break;
                }
            }
            else
            { // x > 5e-4, compute sum4 and sum5 separately
                expx2 = std::exp(-x * x);
                for (int n = 1; 1; ++n)
                {
                    const double coef = std::exp(-a2 * (n * n)) * expx2 / (a2 * (n * n) + y * y);
                    prod2ax *= exp2ax;
                    prodm2ax *= expm2ax;
                    sum1 += coef;
                    sum2 += coef * prodm2ax;
                    sum4 += (coef * prodm2ax) * (a * n);
                    sum3 += coef * prod2ax;
                    sum5 += (coef * prod2ax) * (a * n);
                    // test convergence via sum5, since this sum has the slowest decay
                    if ((coef * prod2ax) * (a * n) < relerr * sum5)
                        break;
                }
            }
        }
        const double expx2erfcxy = // avoid spurious overflow for large negative y
            y > -6                 // for y < -6, erfcx(y) = 2*exp(y*y) to double precision
                ? expx2 * erfcx(y)
                : 2 * std::exp(y * y - x * x);
        if (y > 5)
        { // imaginary terms cancel
            const double sinxy = std::sin(x * y);
            ret = (expx2erfcxy - c * y * sum1) * cos(2 * x * y) + (c * x * expx2) * sinxy * sinc(x * y, sinxy);
        }
        else
        {
            double xs = std::real(z);
            const double sinxy = std::sin(xs * y);
            const double sin2xy = std::sin(2 * xs * y), cos2xy = std::cos(2 * xs * y);
            const double coef1 = expx2erfcxy - c * y * sum1;
            const double coef2 = c * xs * expx2;
            ret = std::complex<double>(coef1 * cos2xy + coef2 * sinxy * sinc(xs * y, sinxy),
                                       coef2 * sinc(2 * xs * y, sin2xy) - coef1 * sin2xy);
        }
    }
    else
    { // x large: only sum3 & sum5 contribute (see above note)
        if (std::isnan(x))
            return std::complex<double>(x, x);
        if (std::isnan(y))
            return std::complex<double>(y, y);

        ret = std::exp(-x * x); // |y| < 1e-10, so we only need exp(-x*x) term
        // (round instead of ceil as in original paper; note that x/a > 1 here)
        double n0 = std::floor(x / a + 0.5); // sum in both directions, starting at n0
        double dx = a * n0 - x;
        sum3 = std::exp(-dx * dx) / (a2 * (n0 * n0) + y * y);
        sum5 = a * n0 * sum3;
        double exp1 = std::exp(4 * a * dx), exp1dn = 1;
        int dn;
        for (dn = 1; n0 - dn > 0; ++dn)
        { // loop over n0-dn and n0+dn terms
            double np = n0 + dn, nm = n0 - dn;
            double tp = exp(-(a * dn + dx) * (a * dn + dx));
            double tm = tp * (exp1dn *= exp1); // trick to get tm from tp
            tp /= (a2 * (np * np) + y * y);
            tm /= (a2 * (nm * nm) + y * y);
            sum3 += tp + tm;
            sum5 += a * (np * tp + nm * tm);
            if (a * (np * tp + nm * tm) < relerr * sum5)
                return ret
                       + std::complex<double>((0.5 * c) * y * (sum2 + sum3),
                                              (0.5 * c) * copysign(sum5 - sum4, std::real(z)));
            ;
        }
        while (1)
        { // loop over n0+dn terms only (since n0-dn <= 0)
            double np = n0 + dn++;
            double tp = std::exp(-(a * dn + dx) * (a * dn + dx)) / (a2 * (np * np) + y * y);
            sum3 += tp;
            sum5 += a * np * tp;
            if (a * np * tp < relerr * sum5)
                return ret
                       + std::complex<double>((0.5 * c) * y * (sum2 + sum3),
                                              (0.5 * c) * copysign(sum5 - sum4, std::real(z)));
            ;
        }
    }
    return ret + std::complex<double>((0.5 * c) * y * (sum2 + sum3), (0.5 * c) * copysign(sum5 - sum4, std::real(z)));
    ;
}

double ErrorFunc::scaled_w_im(double x)
{
    if (x >= 0)
    {
        if (x > 45)
        {                                                        // continued-fraction expansion is faster
            const double ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
            if (x > 5e7)                                         // 1-term expansion, important to avoid overflow
                return ispi / x;
            /* 5-term expansion (rely on compiler for CSE), simplified from:
                      ispi / (x-0.5/(x-1/(x-1.5/(x-2/x))))  */
            return ispi * ((x * x) * (x * x - 4.5) + 2) / (x * ((x * x) * (x * x - 5) + 3.75));
        }
        return w_im_y100(100 / (1 + x), x);
    }
    else
    { // = -FADDEEVA(w_im)(-x)
        if (x < -45)
        {                                                        // continued-fraction expansion is faster
            const double ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
            if (x < -5e7)                                        // 1-term expansion, important to avoid overflow
                return ispi / x;
            /* 5-term expansion (rely on compiler for CSE), simplified from:
                      ispi / (x-0.5/(x-1/(x-1.5/(x-2/x))))  */
            return ispi * ((x * x) * (x * x - 4.5) + 2) / (x * ((x * x) * (x * x - 5) + 3.75));
        }
        return -w_im_y100(100 / (1 - x), -x);
    }
}

std::complex<double> ErrorFunc::erf(std::complex<double> z, double relerr)
{
    double x = std::real(z), y = std::imag(z);

    if (y == 0)
        return std::complex<double>(std::erf(x),
                                    y); // preserve sign of 0
    if (x == 0)                         // handle separately for speed & handling of y = Inf or NaN
        return std::complex<double>(x,  // preserve sign of 0
                                    y * y > 720 ? (y > 0 ? Inf : -Inf) : std::exp(y * y) * scaled_w_im(y));

    double mRe_z2 = (y - x) * (x + y); // Re(-z^2), being careful of overflow
    double mIm_z2 = -2 * x * y;        // Im(-z^2)
    if (mRe_z2 < -750)                 // underflow
        return (x >= 0 ? 1.0 : -1.0);

    // Use Taylor series for small |z|, to avoid cancellation inaccuracy
    //   erf(z) = 2/sqrt(pi) * z * (1 - z^2/3 + z^4/10 - z^6/42 + z^8/216 + ...)
    auto taylor = [&]() -> std::complex<double> {
        std::complex<double> mz2 = std::complex<double>(mRe_z2, mIm_z2); // -z^2
        return z
               * (1.1283791670955125739
                  + mz2
                        * (0.37612638903183752464
                           + mz2
                                 * (0.11283791670955125739
                                    + mz2 * (0.026866170645131251760 + mz2 * 0.0052239776254421878422))));
    };

    /* for small |x| and small |xy|,
       use Taylor series to avoid cancellation inaccuracy:
         erf(x+iy) = erf(iy)
            + 2*exp(y^2)/sqrt(pi) *
              [ x * (1 - x^2 * (1+2y^2)/3 + x^4 * (3+12y^2+4y^4)/30 + ...
                - i * x^2 * y * (1 - x^2 * (3+2y^2)/6 + ...) ]
       where:
          erf(iy) = exp(y^2) * Im[w(y)]
    */
    auto taylor_erfi = [&]() -> std::complex<double> {
        double x2 = x * x, y2 = y * y;
        double expy2 = std::exp(y2);
        return std::complex<double>(
            expy2 * x
                * (1.1283791670955125739 - x2 * (0.37612638903183752464 + 0.75225277806367504925 * y2)
                   + x2 * x2 * (0.11283791670955125739 + y2 * (0.45135166683820502956 + 0.15045055561273500986 * y2))),
            expy2
                * (scaled_w_im(y)
                   - x2 * y * (1.1283791670955125739 - x2 * (0.56418958354775628695 + 0.37612638903183752464 * y2))));
    };

    /* Handle positive and negative x via different formulas,
       using the mirror symmetries of w, to avoid overflow/underflow
       problems from multiplying exponentially large and small quantities. */
    if (x >= 0)
    {
        if (x < 8e-2)
        {
            if (std::fabs(y) < 1e-2)
                return taylor();
            else if (std::fabs(mIm_z2) < 5e-3 && x < 5e-3)
                return taylor_erfi();
        }
        /* don't use complex exp function, since that will produce spurious NaN
           values when multiplying w in an overflow situation. */
        return 1.0
               - std::exp(mRe_z2)
                     * (std::complex<double>(std::cos(mIm_z2), std::sin(mIm_z2))
                        * scaled_w(std::complex<double>(-y, x), relerr));
    }
    else
    { // x < 0
        if (x > -8e-2)
        { // duplicate from above to avoid fabs(x) call
            if (std::fabs(y) < 1e-2)
                return taylor();
            else if (std::fabs(mIm_z2) < 5e-3 && x > -5e-3)
                return taylor_erfi();
        }
        else if (std::isnan(x))
            return std::complex<double>(NaN, y == 0 ? 0 : NaN);
        /* don't use complex exp function, since that will produce spurious NaN
           values when multiplying w in an overflow situation. */
        return std::exp(mRe_z2)
                   * (std::complex<double>(std::cos(mIm_z2), std::sin(mIm_z2))
                      * scaled_w(std::complex<double>(y, -x), relerr))
               - 1.0;
    }
}

// erfi(x) = -i erf(ix)
double ErrorFunc::erfi(double x)
{
    return x * x > 720 ? (x > 0 ? Inf : -Inf) : std::exp(x * x) * scaled_w_im(x);
}

// erfi(z) = -i erf(iz)
std::complex<double> ErrorFunc::erfi(std::complex<double> z, double relerr)
{
    std::complex<double> e = erf(std::complex<double>(-std::imag(z), std::real(z)), relerr);
    return std::complex<double>(std::imag(e), -std::real(e));
}

double ErrorFunc::erfcx(double x) // exp(z^2) erfc(z)
{
    return std::exp(x * x) * std::erfc(x);
}

std::complex<double> ErrorFunc::erfcx(std::complex<double> z, double relerr) // exp(z^2) erfc(z)
{
    return scaled_w(std::complex<double>(-std::imag(z), std::real(z)), relerr);
}

// erfc(z) = 1 - erf(z)
std::complex<double> ErrorFunc::erfc(std::complex<double> z, double relerr)
{
    double x = std::real(z), y = std::imag(z);

    if (x == 0.)
        return std::complex<double>(1,
                                    /* handle y -> Inf limit manually, since
                                       exp(y^2) -> Inf but Im[w(y)] -> 0, so
                                       IEEE will give us a NaN when it should be Inf */
                                    y * y > 720 ? (y > 0 ? -Inf : Inf) : -std::exp(y * y) * scaled_w_im(y));
    if (y == 0.)
    {
        if (x * x > 750) // underflow
            return std::complex<double>(x >= 0 ? 0.0 : 2.0,
                                        -y); // preserve sign of 0
        return std::complex<double>(x >= 0 ? std::exp(-x * x) * erfcx(x) : 2. - std::exp(-x * x) * erfcx(-x),
                                    -y); // preserve sign of zero
    }

    double mRe_z2 = (y - x) * (x + y); // Re(-z^2), being careful of overflow
    double mIm_z2 = -2 * x * y;        // Im(-z^2)
    if (mRe_z2 < -750)                 // underflow
        return (x >= 0 ? 0.0 : 2.0);

    if (x >= 0)
        return std::exp(std::complex<double>(mRe_z2, mIm_z2)) * scaled_w(std::complex<double>(-y, x), relerr);
    else
        return 2.0 - std::exp(std::complex<double>(mRe_z2, mIm_z2)) * scaled_w(std::complex<double>(y, -x), relerr);
}

double ErrorFunc::w_im_y100(double y100, double x)
{
    switch ((int)y100)
    {
    case 0: {
        double t = 2 * y100 - 1;
        return 0.28351593328822191546e-2
               + (0.28494783221378400759e-2
                  + (0.14427470563276734183e-4
                     + (0.10939723080231588129e-6
                        + (0.92474307943275042045e-9
                           + (0.89128907666450075245e-11 + 0.92974121935111111110e-13 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 1: {
        double t = 2 * y100 - 3;
        return 0.85927161243940350562e-2
               + (0.29085312941641339862e-2
                  + (0.15106783707725582090e-4
                     + (0.11716709978531327367e-6
                        + (0.10197387816021040024e-8
                           + (0.10122678863073360769e-10 + 0.10917479678400000000e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 2: {
        double t = 2 * y100 - 5;
        return 0.14471159831187703054e-1
               + (0.29703978970263836210e-2
                  + (0.15835096760173030976e-4
                     + (0.12574803383199211596e-6
                        + (0.11278672159518415848e-8
                           + (0.11547462300333495797e-10 + 0.12894535335111111111e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 3: {
        double t = 2 * y100 - 7;
        return 0.20476320420324610618e-1
               + (0.30352843012898665856e-2
                  + (0.16617609387003727409e-4
                     + (0.13525429711163116103e-6
                        + (0.12515095552507169013e-8
                           + (0.13235687543603382345e-10 + 0.15326595042666666667e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 4: {
        double t = 2 * y100 - 9;
        return 0.26614461952489004566e-1
               + (0.31034189276234947088e-2
                  + (0.17460268109986214274e-4
                     + (0.14582130824485709573e-6
                        + (0.13935959083809746345e-8
                           + (0.15249438072998932900e-10 + 0.18344741882133333333e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 5: {
        double t = 2 * y100 - 11;
        return 0.32892330248093586215e-1
               + (0.31750557067975068584e-2
                  + (0.18369907582308672632e-4
                     + (0.15761063702089457882e-6
                        + (0.15577638230480894382e-8
                           + (0.17663868462699097951e-10
                              + (0.22126732680711111111e-12 + 0.30273474177737853668e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 6: {
        double t = 2 * y100 - 13;
        return 0.39317207681134336024e-1
               + (0.32504779701937539333e-2
                  + (0.19354426046513400534e-4
                     + (0.17081646971321290539e-6
                        + (0.17485733959327106250e-8
                           + (0.20593687304921961410e-10
                              + (0.26917401949155555556e-12 + 0.38562123837725712270e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 7: {
        double t = 2 * y100 - 15;
        return 0.45896976511367738235e-1
               + (0.33300031273110976165e-2
                  + (0.20423005398039037313e-4
                     + (0.18567412470376467303e-6
                        + (0.19718038363586588213e-8
                           + (0.24175006536781219807e-10
                              + (0.33059982791466666666e-12 + 0.49756574284439426165e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 8: {
        double t = 2 * y100 - 17;
        return 0.52640192524848962855e-1
               + (0.34139883358846720806e-2
                  + (0.21586390240603337337e-4
                     + (0.20247136501568904646e-6
                        + (0.22348696948197102935e-8
                           + (0.28597516301950162548e-10
                              + (0.41045502119111111110e-12 + 0.65151614515238361946e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 9: {
        double t = 2 * y100 - 19;
        return 0.59556171228656770456e-1
               + (0.35028374386648914444e-2
                  + (0.22857246150998562824e-4
                     + (0.22156372146525190679e-6
                        + (0.25474171590893813583e-8
                           + (0.34122390890697400584e-10
                              + (0.51593189879111111110e-12 + 0.86775076853908006938e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 10: {
        double t = 2 * y100 - 21;
        return 0.66655089485108212551e-1
               + (0.35970095381271285568e-2
                  + (0.24250626164318672928e-4
                     + (0.24339561521785040536e-6
                        + (0.29221990406518411415e-8
                           + (0.41117013527967776467e-10
                              + (0.65786450716444444445e-12 + 0.11791885745450623331e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 11: {
        double t = 2 * y100 - 23;
        return 0.73948106345519174661e-1
               + (0.36970297216569341748e-2
                  + (0.25784588137312868792e-4
                     + (0.26853012002366752770e-6
                        + (0.33763958861206729592e-8
                           + (0.50111549981376976397e-10
                              + (0.85313857496888888890e-12 + 0.16417079927706899860e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 12: {
        double t = 2 * y100 - 25;
        return 0.81447508065002963203e-1
               + (0.38035026606492705117e-2
                  + (0.27481027572231851896e-4
                     + (0.29769200731832331364e-6
                        + (0.39336816287457655076e-8
                           + (0.61895471132038157624e-10
                              + (0.11292303213511111111e-11 + 0.23558532213703884304e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 13: {
        double t = 2 * y100 - 27;
        return 0.89166884027582716628e-1
               + (0.39171301322438946014e-2
                  + (0.29366827260422311668e-4
                     + (0.33183204390350724895e-6
                        + (0.46276006281647330524e-8
                           + (0.77692631378169813324e-10
                              + (0.15335153258844444444e-11 + 0.35183103415916026911e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 14: {
        double t = 2 * y100 - 29;
        return 0.97121342888032322019e-1
               + (0.40387340353207909514e-2
                  + (0.31475490395950776930e-4
                     + (0.37222714227125135042e-6
                        + (0.55074373178613809996e-8
                           + (0.99509175283990337944e-10
                              + (0.21552645758222222222e-11 + 0.55728651431872687605e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 15: {
        double t = 2 * y100 - 31;
        return 0.10532778218603311137e0
               + (0.41692873614065380607e-2
                  + (0.33849549774889456984e-4
                     + (0.42064596193692630143e-6
                        + (0.66494579697622432987e-8
                           + (0.13094103581931802337e-9
                              + (0.31896187409777777778e-11 + 0.97271974184476560742e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 16: {
        double t = 2 * y100 - 33;
        return 0.11380523107427108222e0
               + (0.43099572287871821013e-2
                  + (0.36544324341565929930e-4
                     + (0.47965044028581857764e-6
                        + (0.81819034238463698796e-8
                           + (0.17934133239549647357e-9
                              + (0.50956666166186293627e-11
                                 + (0.18850487318190638010e-12 + 0.79697813173519853340e-14 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 17: {
        double t = 2 * y100 - 35;
        return 0.12257529703447467345e0
               + (0.44621675710026986366e-2
                  + (0.39634304721292440285e-4
                     + (0.55321553769873381819e-6
                        + (0.10343619428848520870e-7
                           + (0.26033830170470368088e-9
                              + (0.87743837749108025357e-11
                                 + (0.34427092430230063401e-12 + 0.10205506615709843189e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 18: {
        double t = 2 * y100 - 37;
        return 0.13166276955656699478e0
               + (0.46276970481783001803e-2
                  + (0.43225026380496399310e-4
                     + (0.64799164020016902656e-6
                        + (0.13580082794704641782e-7
                           + (0.39839800853954313927e-9
                              + (0.14431142411840000000e-10 + 0.42193457308830027541e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 19: {
        double t = 2 * y100 - 39;
        return 0.14109647869803356475e0
               + (0.48088424418545347758e-2
                  + (0.47474504753352150205e-4
                     + (0.77509866468724360352e-6
                        + (0.18536851570794291724e-7
                           + (0.60146623257887570439e-9
                              + (0.18533978397305276318e-10
                                 + (0.41033845938901048380e-13 - 0.46160680279304825485e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 20: {
        double t = 2 * y100 - 41;
        return 0.15091057940548936603e0
               + (0.50086864672004685703e-2
                  + (0.52622482832192230762e-4
                     + (0.95034664722040355212e-6
                        + (0.25614261331144718769e-7
                           + (0.80183196716888606252e-9
                              + (0.12282524750534352272e-10
                                 + (-0.10531774117332273617e-11 - 0.86157181395039646412e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 21: {
        double t = 2 * y100 - 43;
        return 0.16114648116017010770e0
               + (0.52314661581655369795e-2
                  + (0.59005534545908331315e-4
                     + (0.11885518333915387760e-5
                        + (0.33975801443239949256e-7
                           + (0.82111547144080388610e-9
                              + (-0.12357674017312854138e-10
                                 + (-0.24355112256914479176e-11 - 0.75155506863572930844e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 22: {
        double t = 2 * y100 - 45;
        return 0.17185551279680451144e0
               + (0.54829002967599420860e-2
                  + (0.67013226658738082118e-4
                     + (0.14897400671425088807e-5
                        + (0.40690283917126153701e-7
                           + (0.44060872913473778318e-9
                              + (-0.52641873433280000000e-10 - 0.30940587864543343124e-11 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 23: {
        double t = 2 * y100 - 47;
        return 0.18310194559815257381e0
               + (0.57701559375966953174e-2
                  + (0.76948789401735193483e-4
                     + (0.18227569842290822512e-5
                        + (0.41092208344387212276e-7
                           + (-0.44009499965694442143e-9
                              + (-0.92195414685628803451e-10
                                 + (-0.22657389705721753299e-11 + 0.10004784908106839254e-12 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 24: {
        double t = 2 * y100 - 49;
        return 0.19496527191546630345e0
               + (0.61010853144364724856e-2
                  + (0.88812881056342004864e-4
                     + (0.21180686746360261031e-5
                        + (0.30652145555130049203e-7
                           + (-0.16841328574105890409e-8
                              + (-0.11008129460612823934e-9
                                 + (-0.12180794204544515779e-12 + 0.15703325634590334097e-12 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 25: {
        double t = 2 * y100 - 51;
        return 0.20754006813966575720e0
               + (0.64825787724922073908e-2
                  + (0.10209599627522311893e-3
                     + (0.22785233392557600468e-5
                        + (0.73495224449907568402e-8
                           + (-0.29442705974150112783e-8
                              + (-0.94082603434315016546e-10
                                 + (0.23609990400179321267e-11 + 0.14141908654269023788e-12 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 26: {
        double t = 2 * y100 - 53;
        return 0.22093185554845172146e0
               + (0.69182878150187964499e-2
                  + (0.11568723331156335712e-3
                     + (0.22060577946323627739e-5
                        + (-0.26929730679360840096e-7
                           + (-0.38176506152362058013e-8
                              + (-0.47399503861054459243e-10
                                 + (0.40953700187172127264e-11 + 0.69157730376118511127e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 27: {
        double t = 2 * y100 - 55;
        return 0.23524827304057813918e0
               + (0.74063350762008734520e-2
                  + (0.12796333874615790348e-3
                     + (0.18327267316171054273e-5
                        + (-0.66742910737957100098e-7
                           + (-0.40204740975496797870e-8
                              + (0.14515984139495745330e-10
                                 + (0.44921608954536047975e-11 - 0.18583341338983776219e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 28: {
        double t = 2 * y100 - 57;
        return 0.25058626331812744775e0
               + (0.79377285151602061328e-2
                  + (0.13704268650417478346e-3
                     + (0.11427511739544695861e-5
                        + (-0.10485442447768377485e-6
                           + (-0.34850364756499369763e-8
                              + (0.72656453829502179208e-10
                                 + (0.36195460197779299406e-11 - 0.84882136022200714710e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 29: {
        double t = 2 * y100 - 59;
        return 0.26701724900280689785e0
               + (0.84959936119625864274e-2
                  + (0.14112359443938883232e-3
                     + (0.17800427288596909634e-6
                        + (-0.13443492107643109071e-6
                           + (-0.23512456315677680293e-8
                              + (0.11245846264695936769e-9
                                 + (0.19850501334649565404e-11 - 0.11284666134635050832e-12 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 30: {
        double t = 2 * y100 - 61;
        return 0.28457293586253654144e0
               + (0.90581563892650431899e-2
                  + (0.13880520331140646738e-3
                     + (-0.97262302362522896157e-6
                        + (-0.15077100040254187366e-6
                           + (-0.88574317464577116689e-9
                              + (0.12760311125637474581e-9
                                 + (0.20155151018282695055e-12 - 0.10514169375181734921e-12 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 31: {
        double t = 2 * y100 - 63;
        return 0.30323425595617385705e0
               + (0.95968346790597422934e-2
                  + (0.12931067776725883939e-3
                     + (-0.21938741702795543986e-5
                        + (-0.15202888584907373963e-6
                           + (0.61788350541116331411e-9
                              + (0.11957835742791248256e-9
                                 + (-0.12598179834007710908e-11 - 0.75151817129574614194e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 32: {
        double t = 2 * y100 - 65;
        return 0.32292521181517384379e0
               + (0.10082957727001199408e-1
                  + (0.11257589426154962226e-3
                     + (-0.33670890319327881129e-5
                        + (-0.13910529040004008158e-6
                           + (0.19170714373047512945e-8
                              + (0.94840222377720494290e-10
                                 + (-0.21650018351795353201e-11 - 0.37875211678024922689e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 33: {
        double t = 2 * y100 - 67;
        return 0.34351233557911753862e0
               + (0.10488575435572745309e-1
                  + (0.89209444197248726614e-4
                     + (-0.43893459576483345364e-5
                        + (-0.11488595830450424419e-6
                           + (0.28599494117122464806e-8
                              + (0.61537542799857777779e-10 - 0.24935749227658002212e-11 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 34: {
        double t = 2 * y100 - 69;
        return 0.36480946642143669093e0
               + (0.10789304203431861366e-1
                  + (0.60357993745283076834e-4
                     + (-0.51855862174130669389e-5
                        + (-0.83291664087289801313e-7
                           + (0.33898011178582671546e-8
                              + (0.27082948188277716482e-10
                                 + (-0.23603379397408694974e-11 + 0.19328087692252869842e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 35: {
        double t = 2 * y100 - 71;
        return 0.38658679935694939199e0
               + (0.10966119158288804999e-1
                  + (0.27521612041849561426e-4
                     + (-0.57132774537670953638e-5
                        + (-0.48404772799207914899e-7
                           + (0.35268354132474570493e-8
                              + (-0.32383477652514618094e-11
                                 + (-0.19334202915190442501e-11 + 0.32333189861286460270e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 36: {
        double t = 2 * y100 - 73;
        return 0.40858275583808707870e0
               + (0.11006378016848466550e-1
                  + (-0.76396376685213286033e-5
                     + (-0.59609835484245791439e-5
                        + (-0.13834610033859313213e-7
                           + (0.33406952974861448790e-8
                              + (-0.26474915974296612559e-10
                                 + (-0.13750229270354351983e-11 + 0.36169366979417390637e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 37: {
        double t = 2 * y100 - 75;
        return 0.43051714914006682977e0
               + (0.10904106549500816155e-1
                  + (-0.43477527256787216909e-4
                     + (-0.59429739547798343948e-5
                        + (0.17639200194091885949e-7
                           + (0.29235991689639918688e-8
                              + (-0.41718791216277812879e-10
                                 + (-0.81023337739508049606e-12 + 0.33618915934461994428e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 38: {
        double t = 2 * y100 - 77;
        return 0.45210428135559607406e0
               + (0.10659670756384400554e-1
                  + (-0.78488639913256978087e-4
                     + (-0.56919860886214735936e-5
                        + (0.44181850467477733407e-7
                           + (0.23694306174312688151e-8
                              + (-0.49492621596685443247e-10
                                 + (-0.31827275712126287222e-12 + 0.27494438742721623654e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 39: {
        double t = 2 * y100 - 79;
        return 0.47306491195005224077e0
               + (0.10279006119745977570e-1
                  + (-0.11140268171830478306e-3
                     + (-0.52518035247451432069e-5
                        + (0.64846898158889479518e-7
                           + (0.17603624837787337662e-8
                              + (-0.51129481592926104316e-10
                                 + (0.62674584974141049511e-13 + 0.20055478560829935356e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 40: {
        double t = 2 * y100 - 81;
        return 0.49313638965719857647e0
               + (0.97725799114772017662e-2
                  + (-0.14122854267291533334e-3
                     + (-0.46707252568834951907e-5
                        + (0.79421347979319449524e-7
                           + (0.11603027184324708643e-8
                              + (-0.48269605844397175946e-10
                                 + (0.32477251431748571219e-12 + 0.12831052634143527985e-13 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 41: {
        double t = 2 * y100 - 83;
        return 0.51208057433416004042e0
               + (0.91542422354009224951e-2
                  + (-0.16726530230228647275e-3
                     + (-0.39964621752527649409e-5
                        + (0.88232252903213171454e-7
                           + (0.61343113364949928501e-9
                              + (-0.42516755603130443051e-10
                                 + (0.47910437172240209262e-12 + 0.66784341874437478953e-14 * t) * t)
                                    * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 42: {
        double t = 2 * y100 - 85;
        return 0.52968945458607484524e0
               + (0.84400880445116786088e-2
                  + (-0.18908729783854258774e-3
                     + (-0.32725905467782951931e-5
                        + (0.91956190588652090659e-7
                           + (0.14593989152420122909e-9
                              + (-0.35239490687644444445e-10 + 0.54613829888448694898e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 43: {
        double t = 2 * y100 - 87;
        return 0.54578857454330070965e0
               + (0.76474155195880295311e-2
                  + (-0.20651230590808213884e-3
                     + (-0.25364339140543131706e-5
                        + (0.91455367999510681979e-7
                           + (-0.23061359005297528898e-9
                              + (-0.27512928625244444444e-10 + 0.54895806008493285579e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 44: {
        double t = 2 * y100 - 89;
        return 0.56023851910298493910e0
               + (0.67938321739997196804e-2
                  + (-0.21956066613331411760e-3
                     + (-0.18181127670443266395e-5
                        + (0.87650335075416845987e-7
                           + (-0.51548062050366615977e-9
                              + (-0.20068462174044444444e-10 + 0.50912654909758187264e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 45: {
        double t = 2 * y100 - 91;
        return 0.57293478057455721150e0
               + (0.58965321010394044087e-2
                  + (-0.22841145229276575597e-3
                     + (-0.11404605562013443659e-5
                        + (0.81430290992322326296e-7
                           + (-0.71512447242755357629e-9
                              + (-0.13372664928000000000e-10 + 0.44461498336689298148e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 46: {
        double t = 2 * y100 - 93;
        return 0.58380635448407827360e0
               + (0.49717469530842831182e-2
                  + (-0.23336001540009645365e-3
                     + (-0.51952064448608850822e-6
                        + (0.73596577815411080511e-7
                           + (-0.84020916763091566035e-9
                              + (-0.76700972702222222221e-11 + 0.36914462807972467044e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 47: {
        double t = 2 * y100 - 95;
        return 0.59281340237769489597e0
               + (0.40343592069379730568e-2
                  + (-0.23477963738658326185e-3
                     + (0.34615944987790224234e-7
                        + (0.64832803248395814574e-7
                           + (-0.90329163587627007971e-9
                              + (-0.30421940400000000000e-11 + 0.29237386653743536669e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 48: {
        double t = 2 * y100 - 97;
        return 0.59994428743114271918e0
               + (0.30976579788271744329e-2
                  + (-0.23308875765700082835e-3
                     + (0.51681681023846925160e-6
                        + (0.55694594264948268169e-7
                           + (-0.91719117313243464652e-9
                              + (0.53982743680000000000e-12 + 0.22050829296187771142e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 49: {
        double t = 2 * y100 - 99;
        return 0.60521224471819875444e0
               + (0.21732138012345456060e-2
                  + (-0.22872428969625997456e-3
                     + (0.92588959922653404233e-6
                        + (0.46612665806531930684e-7
                           + (-0.89393722514414153351e-9
                              + (0.31718550353777777778e-11 + 0.15705458816080549117e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 50: {
        double t = 2 * y100 - 101;
        return 0.60865189969791123620e0
               + (0.12708480848877451719e-2
                  + (-0.22212090111534847166e-3
                     + (0.12636236031532793467e-5
                        + (0.37904037100232937574e-7
                           + (-0.84417089968101223519e-9
                              + (0.49843180828444444445e-11 + 0.10355439441049048273e-12 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 51: {
        double t = 2 * y100 - 103;
        return 0.61031580103499200191e0
               + (0.39867436055861038223e-3
                  + (-0.21369573439579869291e-3
                     + (0.15339402129026183670e-5
                        + (0.29787479206646594442e-7
                           + (-0.77687792914228632974e-9
                              + (0.61192452741333333334e-11 + 0.60216691829459295780e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 52: {
        double t = 2 * y100 - 105;
        return 0.61027109047879835868e0
               + (-0.43680904508059878254e-3
                  + (-0.20383783788303894442e-3
                     + (0.17421743090883439959e-5
                        + (0.22400425572175715576e-7
                           + (-0.69934719320045128997e-9
                              + (0.67152759655111111110e-11 + 0.26419960042578359995e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 53: {
        double t = 2 * y100 - 107;
        return 0.60859639489217430521e0
               + (-0.12305921390962936873e-2
                  + (-0.19290150253894682629e-3
                     + (0.18944904654478310128e-5
                        + (0.15815530398618149110e-7
                           + (-0.61726850580964876070e-9 + 0.68987888999111111110e-11 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 54: {
        double t = 2 * y100 - 109;
        return 0.60537899426486075181e0
               + (-0.19790062241395705751e-2
                  + (-0.18120271393047062253e-3
                     + (0.19974264162313241405e-5
                        + (0.10055795094298172492e-7
                           + (-0.53491997919318263593e-9
                              + (0.67794550295111111110e-11 - 0.17059208095741511603e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 55: {
        double t = 2 * y100 - 111;
        return 0.60071229457904110537e0
               + (-0.26795676776166354354e-2
                  + (-0.16901799553627508781e-3
                     + (0.20575498324332621581e-5
                        + (0.51077165074461745053e-8
                           + (-0.45536079828057221858e-9
                              + (0.64488005516444444445e-11 - 0.29311677573152766338e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 56: {
        double t = 2 * y100 - 113;
        return 0.59469361520112714738e0
               + (-0.33308208190600993470e-2
                  + (-0.15658501295912405679e-3
                     + (0.20812116912895417272e-5
                        + (0.93227468760614182021e-9
                           + (-0.38066673740116080415e-9
                              + (0.59806790359111111110e-11 - 0.36887077278950440597e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 57: {
        double t = 2 * y100 - 115;
        return 0.58742228631775388268e0
               + (-0.39321858196059227251e-2
                  + (-0.14410441141450122535e-3
                     + (0.20743790018404020716e-5
                        + (-0.25261903811221913762e-8
                           + (-0.31212416519526924318e-9
                              + (0.54328422462222222221e-11 - 0.40864152484979815972e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 58: {
        double t = 2 * y100 - 117;
        return 0.57899804200033018447e0
               + (-0.44838157005618913447e-2
                  + (-0.13174245966501437965e-3
                     + (0.20425306888294362674e-5
                        + (-0.53330296023875447782e-8
                           + (-0.25041289435539821014e-9
                              + (0.48490437205333333334e-11 - 0.42162206939169045177e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 59: {
        double t = 2 * y100 - 119;
        return 0.56951968796931245974e0
               + (-0.49864649488074868952e-2
                  + (-0.11963416583477567125e-3
                     + (0.19906021780991036425e-5
                        + (-0.75580140299436494248e-8
                           + (-0.19576060961919820491e-9
                              + (0.42613011928888888890e-11 - 0.41539443304115604377e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 60: {
        double t = 2 * y100 - 121;
        return 0.55908401930063918964e0
               + (-0.54413711036826877753e-2
                  + (-0.10788661102511914628e-3
                     + (0.19229663322982839331e-5
                        + (-0.92714731195118129616e-8
                           + (-0.14807038677197394186e-9
                              + (0.36920870298666666666e-11 - 0.39603726688419162617e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 61: {
        double t = 2 * y100 - 123;
        return 0.54778496152925675315e0
               + (-0.58501497933213396670e-2
                  + (-0.96582314317855227421e-4
                     + (0.18434405235069270228e-5
                        + (-0.10541580254317078711e-7
                           + (-0.10702303407788943498e-9
                              + (0.31563175582222222222e-11 - 0.36829748079110481422e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 62: {
        double t = 2 * y100 - 125;
        return 0.53571290831682823999e0
               + (-0.62147030670760791791e-2
                  + (-0.85782497917111760790e-4
                     + (0.17553116363443470478e-5
                        + (-0.11432547349815541084e-7
                           + (-0.72157091369041330520e-10
                              + (0.26630811607111111111e-11 - 0.33578660425893164084e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 63: {
        double t = 2 * y100 - 127;
        return 0.52295422962048434978e0
               + (-0.65371404367776320720e-2
                  + (-0.75530164941473343780e-4
                     + (0.16613725797181276790e-5
                        + (-0.12003521296598910761e-7
                           + (-0.42929753689181106171e-10
                              + (0.22170894940444444444e-11 - 0.30117697501065110505e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 64: {
        double t = 2 * y100 - 129;
        return 0.50959092577577886140e0
               + (-0.68197117603118591766e-2
                  + (-0.65852936198953623307e-4
                     + (0.15639654113906716939e-5
                        + (-0.12308007991056524902e-7
                           + (-0.18761997536910939570e-10
                              + (0.18198628922666666667e-11 - 0.26638355362285200932e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 65: {
        double t = 2 * y100 - 131;
        return 0.49570040481823167970e0
               + (-0.70647509397614398066e-2
                  + (-0.56765617728962588218e-4
                     + (0.14650274449141448497e-5
                        + (-0.12393681471984051132e-7
                           + (0.92904351801168955424e-12
                              + (0.14706755960177777778e-11 - 0.23272455351266325318e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 66: {
        double t = 2 * y100 - 133;
        return 0.48135536250935238066e0
               + (-0.72746293327402359783e-2
                  + (-0.48272489495730030780e-4
                     + (0.13661377309113939689e-5
                        + (-0.12302464447599382189e-7
                           + (0.16707760028737074907e-10
                              + (0.11672928324444444444e-11 - 0.20105801424709924499e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 67: {
        double t = 2 * y100 - 135;
        return 0.46662374675511439448e0
               + (-0.74517177649528487002e-2
                  + (-0.40369318744279128718e-4
                     + (0.12685621118898535407e-5
                        + (-0.12070791463315156250e-7
                           + (0.29105507892605823871e-10
                              + (0.90653314645333333334e-12 - 0.17189503312102982646e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 68: {
        double t = 2 * y100 - 137;
        return 0.45156879030168268778e0
               + (-0.75983560650033817497e-2
                  + (-0.33045110380705139759e-4
                     + (0.11732956732035040896e-5
                        + (-0.11729986947158201869e-7
                           + (0.38611905704166441308e-10
                              + (0.68468768305777777779e-12 - 0.14549134330396754575e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 69: {
        double t = 2 * y100 - 139;
        return 0.43624909769330896904e0
               + (-0.77168291040309554679e-2
                  + (-0.26283612321339907756e-4
                     + (0.10811018836893550820e-5
                        + (-0.11306707563739851552e-7
                           + (0.45670446788529607380e-10
                              + (0.49782492549333333334e-12 - 0.12191983967561779442e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 70: {
        double t = 2 * y100 - 141;
        return 0.42071877443548481181e0
               + (-0.78093484015052730097e-2
                  + (-0.20064596897224934705e-4
                     + (0.99254806680671890766e-6
                        + (-0.10823412088884741451e-7
                           + (0.50677203326904716247e-10
                              + (0.34200547594666666666e-12 - 0.10112698698356194618e-13 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 71: {
        double t = 2 * y100 - 143;
        return 0.40502758809710844280e0
               + (-0.78780384460872937555e-2
                  + (-0.14364940764532853112e-4
                     + (0.90803709228265217384e-6
                        + (-0.10298832847014466907e-7
                           + (0.53981671221969478551e-10
                              + (0.21342751381333333333e-12 - 0.82975901848387729274e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 72: {
        double t = 2 * y100 - 145;
        return 0.38922115269731446690e0
               + (-0.79249269708242064120e-2
                  + (-0.91595258799106970453e-5
                     + (0.82783535102217576495e-6
                        + (-0.97484311059617744437e-8
                           + (0.55889029041660225629e-10
                              + (0.10851981336888888889e-12 - 0.67278553237853459757e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 73: {
        double t = 2 * y100 - 147;
        return 0.37334112915460307335e0
               + (-0.79519385109223148791e-2
                  + (-0.44219833548840469752e-5
                     + (0.75209719038240314732e-6
                        + (-0.91848251458553190451e-8
                           + (0.56663266668051433844e-10
                              + (0.23995894257777777778e-13 - 0.53819475285389344313e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 74: {
        double t = 2 * y100 - 149;
        return 0.35742543583374223085e0
               + (-0.79608906571527956177e-2
                  + (-0.12530071050975781198e-6
                     + (0.68088605744900552505e-6
                        + (-0.86181844090844164075e-8
                           + (0.56530784203816176153e-10
                              + (-0.43120012248888888890e-13 - 0.42372603392496813810e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 75: {
        double t = 2 * y100 - 151;
        return 0.34150846431979618536e0
               + (-0.79534924968773806029e-2
                  + (0.37576885610891515813e-5
                     + (0.61419263633090524326e-6
                        + (-0.80565865409945960125e-8
                           + (0.55684175248749269411e-10
                              + (-0.95486860764444444445e-13 - 0.32712946432984510595e-14 * t) * t)
                                 * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 76: {
        double t = 2 * y100 - 153;
        return 0.32562129649136346824e0
               + (-0.79313448067948884309e-2
                  + (0.72539159933545300034e-5
                     + (0.55195028297415503083e-6
                        + (-0.75063365335570475258e-8
                           + (0.54281686749699595941e-10 - 0.13545424295111111111e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 77: {
        double t = 2 * y100 - 155;
        return 0.30979191977078391864e0
               + (-0.78959416264207333695e-2
                  + (0.10389774377677210794e-4
                     + (0.49404804463196316464e-6
                        + (-0.69722488229411164685e-8
                           + (0.52469254655951393842e-10 - 0.16507860650666666667e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 78: {
        double t = 2 * y100 - 157;
        return 0.29404543811214459904e0
               + (-0.78486728990364155356e-2
                  + (0.13190885683106990459e-4
                     + (0.44034158861387909694e-6
                        + (-0.64578942561562616481e-8
                           + (0.50354306498006928984e-10 - 0.18614473550222222222e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 79: {
        double t = 2 * y100 - 159;
        return 0.27840427686253660515e0
               + (-0.77908279176252742013e-2
                  + (0.15681928798708548349e-4
                     + (0.39066226205099807573e-6
                        + (-0.59658144820660420814e-8
                           + (0.48030086420373141763e-10 - 0.20018995173333333333e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 80: {
        double t = 2 * y100 - 161;
        return 0.26288838011163800908e0
               + (-0.77235993576119469018e-2
                  + (0.17886516796198660969e-4
                     + (0.34482457073472497720e-6
                        + (-0.54977066551955420066e-8
                           + (0.45572749379147269213e-10 - 0.20852924954666666667e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 81: {
        double t = 2 * y100 - 163;
        return 0.24751539954181029717e0
               + (-0.76480877165290370975e-2
                  + (0.19827114835033977049e-4
                     + (0.30263228619976332110e-6
                        + (-0.50545814570120129947e-8
                           + (0.43043879374212005966e-10 - 0.21228012028444444444e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 82: {
        double t = 2 * y100 - 165;
        return 0.23230087411688914593e0
               + (-0.75653060136384041587e-2
                  + (0.21524991113020016415e-4
                     + (0.26388338542539382413e-6
                        + (-0.46368974069671446622e-8
                           + (0.40492715758206515307e-10 - 0.21238627815111111111e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 83: {
        double t = 2 * y100 - 167;
        return 0.21725840021297341931e0
               + (-0.74761846305979730439e-2
                  + (0.23000194404129495243e-4
                     + (0.22837400135642906796e-6
                        + (-0.42446743058417541277e-8
                           + (0.37958104071765923728e-10 - 0.20963978568888888889e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 84: {
        double t = 2 * y100 - 169;
        return 0.20239979200788191491e0
               + (-0.73815761980493466516e-2
                  + (0.24271552727631854013e-4
                     + (0.19590154043390012843e-6
                        + (-0.38775884642456551753e-8
                           + (0.35470192372162901168e-10 - 0.20470131678222222222e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 85: {
        double t = 2 * y100 - 171;
        return 0.18773523211558098962e0
               + (-0.72822604530339834448e-2
                  + (0.25356688567841293697e-4
                     + (0.16626710297744290016e-6
                        + (-0.35350521468015310830e-8
                           + (0.33051896213898864306e-10 - 0.19811844544000000000e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 86: {
        double t = 2 * y100 - 173;
        return 0.17327341258479649442e0
               + (-0.71789490089142761950e-2
                  + (0.26272046822383820476e-4
                     + (0.13927732375657362345e-6
                        + (-0.32162794266956859603e-8
                           + (0.30720156036105652035e-10 - 0.19034196304000000000e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 87: {
        double t = 2 * y100 - 175;
        return 0.15902166648328672043e0
               + (-0.70722899934245504034e-2
                  + (0.27032932310132226025e-4
                     + (0.11474573347816568279e-6
                        + (-0.29203404091754665063e-8
                           + (0.28487010262547971859e-10 - 0.18174029063111111111e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 88: {
        double t = 2 * y100 - 177;
        return 0.14498609036610283865e0
               + (-0.69628725220045029273e-2
                  + (0.27653554229160596221e-4
                     + (0.92493727167393036470e-7
                        + (-0.26462055548683583849e-8
                           + (0.26360506250989943739e-10 - 0.17261211260444444444e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 89: {
        double t = 2 * y100 - 179;
        return 0.13117165798208050667e0
               + (-0.68512309830281084723e-2
                  + (0.28147075431133863774e-4
                     + (0.72351212437979583441e-7
                        + (-0.23927816200314358570e-8
                           + (0.24345469651209833155e-10 - 0.16319736960000000000e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 90: {
        double t = 2 * y100 - 181;
        return 0.11758232561160626306e0
               + (-0.67378491192463392927e-2
                  + (0.28525664781722907847e-4
                     + (0.54156999310046790024e-7
                        + (-0.21589405340123827823e-8
                           + (0.22444150951727334619e-10 - 0.15368675584000000000e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 91: {
        double t = 2 * y100 - 183;
        return 0.10422112945361673560e0
               + (-0.66231638959845581564e-2
                  + (0.28800551216363918088e-4
                     + (0.37758983397952149613e-7
                        + (-0.19435423557038933431e-8
                           + (0.20656766125421362458e-10 - 0.14422990012444444444e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 92: {
        double t = 2 * y100 - 185;
        return 0.91090275493541084785e-1
               + (-0.65075691516115160062e-2
                  + (0.28982078385527224867e-4
                     + (0.23014165807643012781e-7
                        + (-0.17454532910249875958e-8
                           + (0.18981946442680092373e-10 - 0.13494234691555555556e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 93: {
        double t = 2 * y100 - 187;
        return 0.78191222288771379358e-1
               + (-0.63914190297303976434e-2
                  + (0.29079759021299682675e-4
                     + (0.97885458059415717014e-8
                        + (-0.15635596116134296819e-8
                           + (0.17417110744051331974e-10 - 0.12591151763555555556e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 94: {
        double t = 2 * y100 - 189;
        return 0.65524757106147402224e-1
               + (-0.62750311956082444159e-2
                  + (0.29102328354323449795e-4
                     + (-0.20430838882727954582e-8
                        + (-0.13967781903855367270e-8
                           + (0.15958771833747057569e-10 - 0.11720175765333333333e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 95: {
        double t = 2 * y100 - 191;
        return 0.53091065838453612773e-1
               + (-0.61586898417077043662e-2
                  + (0.29057796072960100710e-4
                     + (-0.12597414620517987536e-7
                        + (-0.12440642607426861943e-8
                           + (0.14602787128447932137e-10 - 0.10885859114666666667e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 96: {
        double t = 2 * y100 - 193;
        return 0.40889797115352738582e-1
               + (-0.60426484889413678200e-2
                  + (0.28953496450191694606e-4
                     + (-0.21982952021823718400e-7
                        + (-0.11044169117553026211e-8
                           + (0.13344562332430552171e-10 - 0.10091231402844444444e-12 * t) * t)
                              * t)
                           * t)
                        * t)
                     * t;
    }
    case 97:
    case 98:
    case 99:
    case 100: { // use Taylor expansion for small x (|x| <= 0.0309...)
        //  (2/sqrt(pi)) * (x - 2/3 x^3  + 4/15 x^5  - 8/105 x^7 + 16/945 x^9)
        double x2 = x * x;
        return x
               * (1.1283791670955125739
                  - x2
                        * (0.75225277806367504925
                           - x2
                                 * (0.30090111122547001970
                                    - x2 * (0.085971746064420005629 - x2 * 0.016931216931216931217))));
    }
    }
    /* Since 0 <= y100 < 101, this is only reached if x is NaN,
       in which case we should return NaN. */
    return NaN;
}

// precomputed table of expa2n2[n-1] = exp(-a2*n*n)
const std::vector<double> ErrorFunc::expa2n2 = {
    7.64405281671221563e-01,
    3.41424527166548425e-01,
    8.91072646929412548e-02,
    1.35887299055460086e-02,
    1.21085455253437481e-03,
    6.30452613933449404e-05,
    1.91805156577114683e-06,
    3.40969447714832381e-08,
    3.54175089099469393e-10,
    2.14965079583260682e-12,
    7.62368911833724354e-15,
    1.57982797110681093e-17,
    1.91294189103582677e-20,
    1.35344656764205340e-23,
    5.59535712428588720e-27,
    1.35164257972401769e-30,
    1.90784582843501167e-34,
    1.57351920291442930e-38,
    7.58312432328032845e-43,
    2.13536275438697082e-47,
    3.51352063787195769e-52,
    3.37800830266396920e-57,
    1.89769439468301000e-62,
    6.22929926072668851e-68,
    1.19481172006938722e-73,
    1.33908181133005953e-79,
    8.76924303483223939e-86,
    3.35555576166254986e-92,
    7.50264110688173024e-99,
    9.80192200745410268e-106,
    7.48265412822268959e-113,
    3.33770122566809425e-120,
    8.69934598159861140e-128,
    1.32486951484088852e-135,
    1.17898144201315253e-143,
    6.13039120236180012e-152,
    1.86258785950822098e-160,
    3.30668408201432783e-169,
    3.43017280887946235e-178,
    2.07915397775808219e-187,
    7.36384545323984966e-197,
    1.52394760394085741e-206,
    1.84281935046532100e-216,
    1.30209553802992923e-226,
    5.37588903521080531e-237,
    1.29689584599763145e-247,
    1.82813078022866562e-258,
    1.50576355348684241e-269,
    7.24692320799294194e-281,
    2.03797051314726829e-292,
    3.34880215927873807e-304,
    0.0 // underflow (also prevents reads past array end, below)
};

} // namespace ModuleBase
