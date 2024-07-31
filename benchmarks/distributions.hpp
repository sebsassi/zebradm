#include <array>
#include <cmath>
#include <numbers>

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

/*
linear combination of a centered wide isotropic Gaussian with shifted narrow isotropic Gaussian.
*/
double shm_plus_stream(double r, double lon, double colat)
{
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

    return 0.8*shm_part + 0.2*stream_part;
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

/*
linear combination of an isotropic and an anisotropic Gaussian.
*/
double shmpp(double r, double lon, double colat)
{
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

    return (1.0 - eta)*shm_part + eta*pp_part;
}