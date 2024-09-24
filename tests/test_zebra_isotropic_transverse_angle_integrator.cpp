#include "zebra_angle_integrator.hpp"

#include <cassert>

[[maybe_unused]] constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

[[maybe_unused]] constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::max(std::fabs(a[0] - b[0]), std::fabs(a[1] - b[1])) < tol;
}

template <typename T, std::size_t N>
double horner(const std::array<T, N>& coeffs, T x)
{
    T res{};
    for (const T& coeff : coeffs | std::views::reverse)
        res = coeff + res*x;
    return res;
}

std::array<double, 2> angle_integrated_const_dist_radon_pair(
    double min_speed, const std::array<double, 3>& boost)
{
    const double boost_speed = length(boost);
    const double v = boost_speed;
    const double v2 = v*v;
    const double w = min_speed;
    const double w2 = w*w;
    const double zmin = std::max(-1.0, -(1.0 + w)/v);
    const double zmax = std::min(1.0, (1.0 - w)/v);

    const std::array<double, 5> trans_coeffs = {
        v2*(1.0 - w2) - w2 + 0.5*(1.0 + w2*w2),
        -w*v*(1.0 + v2 - w2),
        -(1.0/3.0)*(2.0 + v2 - 4.0*w2)*v2,
        w*v2*v,
        0.3*v2*v2
    };

    const std::array<double, 3> nontrans_coeffs = {
        1.0 - w2, -w*v, -(1.0/3.0)*v2
    };

    constexpr double two_pi_sq = 2.0*std::numbers::pi*std::numbers::pi;
    const double trans_res = zmax*horner(trans_coeffs, zmax) - zmin*horner(trans_coeffs, zmin);
    const double nontrans_res = zmax*horner(nontrans_coeffs, zmax) - zmin*horner(nontrans_coeffs, zmin);
    return {two_pi_sq*nontrans_res, two_pi_sq*trans_res};
}

bool test_transverse_angle_integrator_is_correct_for_constant_dist()
{
    std::vector<Vector<double, 3>> boosts = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> min_speeds = {0.0};//, 0.5, 1.0, 1.5};

    std::vector<std::array<double, 2>> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});

    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            reference(i, j) = angle_integrated_const_dist_radon_pair(
                    min_speeds[j], boosts[i]);
        }
    }

    constexpr std::size_t order = 1;
    zest::zt::ZernikeExpansionOrthoGeo distribution(order);
    distribution(0,0,0) = {1.0/std::sqrt(3.0), 0.0};

    std::vector<std::array<double, 2>> test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> test(
            test_buffer.data(), {boosts.size(), min_speeds.size()});

    zebra::IsotropicTransverseAngleIntegrator(order)
        .integrate(distribution, boosts, min_speeds, test);
    
    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
            success = success && is_close(test(i,j), reference(i,j), tol);
    }

    if (!success)
    {
        std::printf("reference\n");
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
            {
                std::printf("{%.16e, %.16e} ", reference(i, j)[0], reference(i, j)[1]);
            }
            std::printf("\n");
        }

        std::printf("\ntest\n");
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
            {
                std::printf("{%.16e, %.16e} ", test(i, j)[0], test(i, j)[1]);
            }
            std::printf("\n");
        }
    }

    return success;
}

int main()
{
    assert(test_transverse_angle_integrator_is_correct_for_constant_dist());
}