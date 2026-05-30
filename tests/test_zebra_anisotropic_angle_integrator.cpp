/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include <cassert>
#include <print>

#include <zest/md_array.hpp>

#include "types.hpp"
#include "zebra_angle_integrator.hpp"

namespace
{

[[maybe_unused]] constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

template <typename T, std::size_t N>
double horner(const std::array<T, N>& coeffs, T x)
{
    T res{};
    for (const T& coeff : coeffs | std::views::reverse)
        res = coeff + res*x;
    return res;
}

[[maybe_unused]] double angle_integrated_const_dist_radon(
    double shell, const zdm::la::Vector<double, 3>& offset)
{
    const double offset_len = zdm::la::length(offset);
    const double v = offset_len;
    const double v2 = v*v;
    const double w = shell;
    const double w2 = w*w;
    const double zmin = std::max(-1.0, -(1.0 + w)/v);
    const double zmax = std::min(1.0, (1.0 - w)/v);

    const std::array<double, 3> coeffs = {
        1.0 - w2, -w*v, -(1.0/3.0)*v2
    };

    constexpr double two_pi_sq = 2.0*std::numbers::pi*std::numbers::pi;
    const double res = zmax*horner(coeffs, zmax) - zmin*horner(coeffs, zmin);
    return two_pi_sq*res;
}

bool test_angle_integrator_is_correct_for_constant_dist_constant_resp()
{
    std::vector<zdm::la::Vector<double, 3>> offsets = {
        {1.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };
    std::vector<double> rotation_angles = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<double> shells = {0.0, 0.5, 1.0, 1.5};

    std::vector<double> reference_buffer(offsets.size()*shells.size());
    zest::DynamicMDArray<double, 2> reference{offsets.size(), shells.size()};

    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            reference(i, j)
                = angle_integrated_const_dist_radon(shells[j], offsets[i]);
        }
    }

    constexpr std::size_t order = 1;
    zdm::ZernikeExpansion<double> distribution{order};
    distribution[0, 0, 0, 0] = std::numbers::inv_sqrt3;
    distribution[0, 0, 0, 1] = 0.0;

    zdm::SHExpansionVector<double> resp{shells.size(), order};
    for (std::size_t i = 0; i < shells.size(); ++i)
        resp[i, 0, 0, 0] = 1.0;

    zest::DynamicMDArray<double, 2> test{offsets.size(), shells.size()};

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::aniso>(order, order)
        .integrate(distribution, resp, offsets, rotation_angles, shells, test);

    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            success = success && is_close(test[i, j], reference[i, j], tol);
    }

    if (!success)
    {
        std::println("reference");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", reference[i, j]);
            }
            std::println("");
        }

        std::println("\ntest");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", test[i, j]);
            }
            std::println("");
        }
    }

    return success;
}

double angle_integrated_radon_shm(
    const zdm::la::Vector<double, 3>& offset, double shell, double disp_speed)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    const double offset_len = zdm::la::length(offset);
    const double erf_part
        = std::erf(std::min(1.0,shell + offset_len)/disp_speed)
        - std::erf((shell - offset_len)/disp_speed);

    const double exp_prefactor
        = (1.0 + offset_len - std::max(1.0 - offset_len, shell));

    const double inv_disp = 1.0/disp_speed;
    const double prefactor = std::numbers::pi*disp_speed*disp_speed/offset_len;

    return (2.0*std::numbers::pi)*prefactor*(0.5*sqrt_pi*disp_speed*erf_part - exp_prefactor*std::exp(-inv_disp*inv_disp));
}

bool test_angle_integrator_is_accurate_for_shm_constant_resp()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const std::array<double, 3>& velocity){
        const double speed = zdm::la::length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    std::vector<zdm::la::Vector<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };
    std::vector<double> rotation_angles = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const std::vector<double> shells = {
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    zest::DynamicMDArray<double, 2> shm_reference{offsets.size(), shells.size()};
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            shm_reference(i, j) = angle_integrated_radon_shm(
                    offsets[i], shells[j], disp_speed);
        }
    }

    zest::DynamicMDArray<double, 2> shm_test{offsets.size(), shells.size()};

    constexpr std::size_t order = 100;
    zdm::ZernikeExpansion<double> distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(order).forward_transform(
                shm_dist, 1.0, order);

    zdm::SHExpansionVector<double> resp{shells.size(), order};
    for (std::size_t i = 0; i < shells.size(); ++i)
        resp[i, 0, 0, 0] = 1.0;

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::aniso>(order, order)
        .integrate(distribution, resp, offsets, rotation_angles, shells, shm_test);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            is_close(shm_test[i, j]/shm_reference[i, j], 1.0, tol);
    }

    if (!success)
    {
        std::println("reference");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", shm_reference[i, j]);
            }
            std::println("");
        }

        std::println("\ntest");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", shm_test[i, j]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", 1.0 - shm_test[i, j]/shm_reference[i, j]);
            }
            std::println("");
        }
    }

    return success;
}

} // namespace

int main()
{
    assert(test_angle_integrator_is_correct_for_constant_dist_constant_resp());
    assert(test_angle_integrator_is_accurate_for_shm_constant_resp());
}
