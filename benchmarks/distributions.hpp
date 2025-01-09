/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#include <array>
#include <cmath>
#include <numbers>

using DistributionSpherical = double(*)(double, double, double);
using DistributionCartesian = double(*)(const std::array<double, 3>&);

double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

/*
anisotropic Gaussian with arbitrary covariance.
*/
double aniso_gaussian(double r, double lon, double colat)
{
    constexpr std::array<std::array<double, 3>, 3> sigma = {
        std::array<double, 3>{3.0, 1.4, 0.5},
        std::array<double, 3>{1.4, 0.3, 2.1},
        std::array<double, 3>{0.5, 2.1, 1.7}
    };
    const std::array<double, 3> x = {
        r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
    return std::exp(-0.5*quadratic_form(sigma, x));
}

double aniso_gaussian(const std::array<double, 3>& x)
{
    constexpr std::array<std::array<double, 3>, 3> sigma = {
        std::array<double, 3>{3.0, 1.4, 0.5},
        std::array<double, 3>{1.4, 0.3, 2.1},
        std::array<double, 3>{0.5, 2.1, 1.7}
    };
    return std::exp(-0.5*quadratic_form(sigma, x));
}

/*
linear combination of four isotropic Gaussians of different means and widths.
*/
double four_gaussians(double r, double lon, double colat)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double ve = 537.0;
    constexpr std::array<std::array<double, 3>, 4> v0 = {
        std::array<double, 3>{0.0, 0.0, -230.0/ve},
        std::array<double, 3>{80.0/ve, 0.0, -80.0/ve},
        std::array<double, 3>{-120.0/ve, -250.0/ve, -150.0/ve},
        std::array<double, 3>{50.0/ve, 30.0/ve, -400.0/ve}
    };
    constexpr std::array<double, 4> v_d = {
        220.0/ve, 70.0/ve, 50.0/ve, 25.0/ve
    };
    constexpr std::array<double, 4> frac = {0.4, 0.3, 0.2, 0.1};
    constexpr std::array<double, 4> norm = {
        frac[0]/(sqrt_pi_3*v_d[0]*v_d[0]*v_d[0]),
        frac[1]/(sqrt_pi_3*v_d[1]*v_d[1]*v_d[1]),
        frac[2]/(sqrt_pi_3*v_d[2]*v_d[2]*v_d[2]),
        frac[3]/(sqrt_pi_3*v_d[3]*v_d[3]*v_d[3]),
    };

    const std::array<double, 3> v = {
        r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
    };
    double res = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const std::array<double, 3> dv = {
            v[0] - v0[i][0], v[1] - v0[i][1], v[2] - v0[i][2]
        };
        res += norm[i]*std::exp(-(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])/(v_d[i]*v_d[i]));
    }
    return res;
}


double four_gaussians(const std::array<double, 3>& v)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double ve = 537.0;
    constexpr std::array<std::array<double, 3>, 4> v0 = {
        std::array<double, 3>{0.0, 0.0, -230.0/ve},
        std::array<double, 3>{80.0/ve, 0.0, -80.0/ve},
        std::array<double, 3>{-120.0/ve, -250.0/ve, -150.0/ve},
        std::array<double, 3>{50.0/ve, 30.0/ve, -400.0/ve}
    };
    constexpr std::array<double, 4> v_d = {
        220.0/ve, 70.0/ve, 50.0/ve, 25.0/ve
    };
    constexpr std::array<double, 4> frac = {0.4, 0.3, 0.2, 0.1};
    constexpr std::array<double, 4> norm = {
        frac[0]/(sqrt_pi_3*v_d[0]*v_d[0]*v_d[0]),
        frac[1]/(sqrt_pi_3*v_d[1]*v_d[1]*v_d[1]),
        frac[2]/(sqrt_pi_3*v_d[2]*v_d[2]*v_d[2]),
        frac[3]/(sqrt_pi_3*v_d[3]*v_d[3]*v_d[3]),
    };

    double res = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const std::array<double, 3> dv = {
            v[0] - v0[i][0], v[1] - v0[i][1], v[2] - v0[i][2]
        };
        res += norm[i]*std::exp(-(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])/(v_d[i]*v_d[i]));
    }
    return res;
}

/*
linear combination of a centered wide isotropic Gaussian with shifted narrow isotropic Gaussian.
*/
double shm_plus_stream(double r, double lon, double colat)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double ve = 544.0;
    constexpr double v0 = 220.0/ve;
    constexpr double v0_sq = v0*v0;
    constexpr double inv_v0_sq = 1.0/(v0_sq);

    constexpr double vs = 25.0/ve;
    constexpr double inv_vs = 1.0/vs;

    constexpr std::array<double, 3> x0 = {50.0/ve, 30.0/ve, -400.0/ve};

    const std::array<double, 3> x = {
        r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
    const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

    const std::array<double, 3> xs = {
        x[0] - x0[0], x[1] - x0[2], x[2] - x0[2]
    };
    const double quad_stream
            = inv_vs*(xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);

    const double shm_part = std::exp(-quad_shm);
    const double stream_part = std::exp(-quad_stream);

    constexpr std::array<double, 2> frac = {0.8, 0.2};
    constexpr std::array<double, 2> norm = {
        frac[0]/(sqrt_pi_3*v0*v0*v0),
        frac[1]/(sqrt_pi_3*vs*vs*vs)
    };

    return norm[0]*shm_part + norm[1]*stream_part;
}

double shm_plus_stream(const std::array<double, 3>& x)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double ve = 544.0;
    constexpr double v0 = 220.0/ve;
    constexpr double v0_sq = v0*v0;
    constexpr double inv_v0_sq = 1.0/(v0_sq);

    constexpr double vs = 25.0/ve;
    constexpr double inv_vs = 1.0/vs;

    constexpr std::array<double, 3> x0 = {50.0/ve, 30.0/ve, -400.0/ve};

    const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

    const std::array<double, 3> xs = {
        x[0] - x0[0], x[1] - x0[2], x[2] - x0[2]
    };
    const double quad_stream
            = inv_vs*(xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);

    const double shm_part = std::exp(-quad_shm);
    const double stream_part = std::exp(-quad_stream);

    constexpr std::array<double, 2> frac = {0.8, 0.2};
    constexpr std::array<double, 2> norm = {
        frac[0]/(sqrt_pi_3*v0*v0*v0),
        frac[1]/(sqrt_pi_3*vs*vs*vs)
    };

    return norm[0]*shm_part + norm[1]*stream_part;
}

/*
anisotropic Gaussian.
*/
double shmpp_aniso(double r, double lon, double colat)
{
    constexpr double beta = 0.9;
    constexpr double v0 = 233.0/580.0;
    constexpr double v0_sq = v0*v0;
    constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
    constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
    constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
    constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
    const std::array<double, 3> x = {
        r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
    const double quad = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);

    return std::exp(-quad);
}

double shmpp_aniso(const std::array<double, 3>& x)
{
    constexpr double beta = 0.9;
    constexpr double v0 = 233.0/580.0;
    constexpr double v0_sq = v0*v0;
    constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
    constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
    constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
    constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
    
    const double quad = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);

    return std::exp(-quad);
}

/*
linear combination of an isotropic and an anisotropic Gaussian.
*/
double shmpp(double r, double lon, double colat)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double eta = 0.3;
    constexpr double beta = 0.9;
    constexpr double v0 = 233.0/580.0;
    constexpr double v0_sq = v0*v0;
    constexpr double inv_v0_sq = 1.0/(v0_sq);
    constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
    constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
    constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
    constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
    const std::array<double, 3> x = {
        r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)};
    const double quad_pp = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);
    const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

    const double shm_part = std::exp(-quad_shm);
    const double pp_part = std::exp(-quad_pp);
    constexpr std::array<double, 2> norm = {
        (1.0 - eta)/(sqrt_pi_3*v0*v0*v0),
        eta/(sqrt_pi_3*sigma_r_sq*sigma_th_sq*sigma_th_sq)
    };

    return norm[0]*shm_part + norm[1]*pp_part;
}

double shmpp(const std::array<double, 3>& x)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    constexpr double sqrt_pi_3 = sqrt_pi*sqrt_pi*sqrt_pi;
    constexpr double eta = 0.3;
    constexpr double beta = 0.9;
    constexpr double v0 = 233.0/580.0;
    constexpr double v0_sq = v0*v0;
    constexpr double inv_v0_sq = 1.0/(v0_sq);
    constexpr double sigma_r_sq = 1.5*v0_sq/(3.0 - 2.0*beta);
    constexpr double sigma_th_sq = sigma_r_sq*(1.0 - beta);
    constexpr double inv_sigma_r_sq = 1.0/sigma_r_sq;
    constexpr double inv_sigma_th_sq = 1.0/sigma_th_sq;
    
    const double quad_pp = inv_sigma_r_sq*x[0]*x[0] + inv_sigma_th_sq*(x[1]*x[1] + x[2]*x[2]);
    const double quad_shm = inv_v0_sq*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

    const double shm_part = std::exp(-quad_shm);
    const double pp_part = std::exp(-quad_pp);
    constexpr std::array<double, 2> norm = {
        (1.0 - eta)/(sqrt_pi_3*v0*v0*v0),
        eta/(sqrt_pi_3*sigma_r_sq*sigma_th_sq*sigma_th_sq)
    };

    return norm[0]*shm_part + norm[1]*pp_part;
}