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
#include "radon_integrator.hpp"
#include "transform_utilities.hpp"
#include "types.hpp"
#include "zebra_angle_integrator.hpp"

#include "coordinate_transforms.hpp"

#include <print>

#include <zest/md_array.hpp>

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) <= 0.5*std::fabs(a + b)*tol + tol;
}

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

template <typename DistType>
[[nodiscard]] bool test_mutual_convergence_iso_iso(
    DistType&& dist, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells)
{
    constexpr double tol = 1.0e-9;

    zest::DynamicMDArray<double, 2> integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            [&](const std::array<double, 3>& v){
                return std::forward<DistType>(dist)(zdm::la::length(v));
            },
            offsets, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<double, 2> transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t order = 200;
    zdm::IsotropicZernikeExpansion<double> distribution
        = zest::zt::IsotropicZernikeTransformerNormalGeo<>(order).forward_transform(
                std::forward<DistType>(dist), 1.0, order);

    zdm::zebra::AngleIntegrator<zdm::DistType::iso, zdm::RespType::iso>(order)
        .integrate(distribution, offsets, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success && is_close(integrator_test[i, j], transformer_test[i, j], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", integrator_test[i, j]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", transformer_test[i, j]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", 1.0 - integrator_test[i, j]/transformer_test[i, j]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType>
[[nodiscard]] bool test_mutual_convergence_aniso_iso(
    DistType&& dist, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells)
{
    constexpr double tol = 1.0e-9;

    zest::DynamicMDArray<double, 2> integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            std::forward<DistType>(dist), offsets, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<double, 2> transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t order = 200;
    zdm::ZernikeExpansion<double> distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(order).forward_transform(
                std::forward<DistType>(dist), 1.0, order);

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso>(order)
        .integrate(distribution, offsets, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success && is_close(integrator_test[i, j], transformer_test[i, j], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", integrator_test[i, j]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", transformer_test[i, j]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", 1.0 - integrator_test[i, j]/transformer_test[i, j]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType>
[[nodiscard]] bool test_mutual_convergence_transverse_iso_iso(
    DistType&& dist, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells)
{
    constexpr double tol = 1.0e-9;

    zest::DynamicMDArray<std::array<double, 2>, 2>
    integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(
        [&](const std::array<double, 3>& v){
            return std::forward<DistType>(dist)(zdm::la::length(v));
        },
        offsets, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<std::array<double, 2>, 2>
    transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t order = 200;
    zdm::IsotropicZernikeExpansion<double> distribution
        = zest::zt::IsotropicZernikeTransformerNormalGeo<>(order).forward_transform(
                std::forward<DistType>(dist), 1.0, order);

    zdm::zebra::TransverseAngleIntegrator<zdm::DistType::iso, zdm::RespType::iso>(order)
        .integrate(distribution, offsets, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success
                && is_close(integrator_test[i, j][0], transformer_test[i, j][0], tol)
                && is_close(integrator_test[i, j][1], transformer_test[i, j][1], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        integrator_test[i, j][0], integrator_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        transformer_test[i, j][0], transformer_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        1.0 - integrator_test[i, j][0]/transformer_test[i, j][0],
                        1.0 - integrator_test[i, j][1]/transformer_test[i, j][1]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType>
[[nodiscard]] bool test_mutual_convergence_transverse_aniso_iso(
    DistType&& dist, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells)
{
    constexpr double tol = 1.0e-9;

    zest::DynamicMDArray<std::array<double, 2>, 2>
    integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(
            std::forward<DistType>(dist), offsets, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<std::array<double, 2>, 2>
    transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t order = 200;
    zdm::ZernikeExpansion<double> distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(order).forward_transform(
                std::forward<DistType>(dist), 1.0, order);

    zdm::zebra::TransverseAngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso>(order)
        .integrate(distribution, offsets, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success
                && is_close(integrator_test[i, j][0], transformer_test[i, j][0], tol)
                && is_close(integrator_test[i, j][1], transformer_test[i, j][1], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        integrator_test[i, j][0], integrator_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        transformer_test[i, j][0], transformer_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        1.0 - integrator_test[i, j][0]/transformer_test[i, j][0],
                        1.0 - integrator_test[i, j][1]/transformer_test[i, j][1]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType, typename RespType>
[[nodiscard]] bool test_mutual_convergence_iso_aniso(
    DistType&& dist, RespType&& resp, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    constexpr double tol = 1.0e-7;

    zest::DynamicMDArray<double, 2> integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            [&](const std::array<double, 3>& v){
                return std::forward<DistType>(dist)(zdm::la::length(v));
            },
            std::forward<RespType>(resp), offsets, rotation_angles, shells, 0.0, tol,
            integrator_test);

    zest::DynamicMDArray<double, 2> transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zdm::IsotropicZernikeExpansion<double> distribution
        = zest::zt::IsotropicZernikeTransformerNormalGeo<>(dist_order).forward_transform(
                std::forward<DistType>(dist), 1.0, dist_order);

    zdm::SHExpansionVector<double> response{shells.size(), resp_order};
    zdm::ResponseTransformer(resp_order)
        .forward_transform(std::forward<RespType>(resp), shells, response);

    zdm::zebra::AngleIntegrator<zdm::DistType::iso, zdm::RespType::aniso>(dist_order, resp_order)
        .integrate(distribution, response, offsets, rotation_angles, shells, transformer_test);


    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success && is_close(integrator_test[i, j], transformer_test[i, j], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", integrator_test[i, j]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", transformer_test[i, j]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", 1.0 - integrator_test[i, j]/transformer_test[i, j]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType, typename RespType>
[[nodiscard]] bool test_mutual_convergence_aniso_aniso(
    DistType&& dist, RespType&& resp, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    constexpr double tol = 1.0e-7;

    zest::DynamicMDArray<double, 2> integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            std::forward<DistType>(dist), std::forward<RespType>(resp),
            offsets, rotation_angles, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<double, 2> transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zdm::ZernikeExpansion<double> distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(dist_order).forward_transform(
                std::forward<DistType>(dist), 1.0, dist_order);

    zdm::SHExpansionVector<double> response{shells.size(), resp_order};
    zdm::ResponseTransformer(resp_order)
        .forward_transform(std::forward<RespType>(resp), shells, response);

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::aniso>(dist_order, resp_order)
        .integrate(distribution, response, offsets, rotation_angles, shells, transformer_test);


    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success && is_close(integrator_test[i, j], transformer_test[i, j], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", integrator_test[i, j]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", transformer_test[i, j]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("{:.16e} ", 1.0 - integrator_test[i, j]/transformer_test[i, j]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType, typename RespType>
[[nodiscard]] bool test_mutual_convergence_transverse_iso_aniso(
    DistType&& dist, RespType&& resp, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    constexpr double tol = 1.0e-7;

    zest::DynamicMDArray<std::array<double, 2>, 2>
    integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(
            [&](const std::array<double, 3>& v){
                return std::forward<DistType>(dist)(zdm::la::length(v));
            },
            std::forward<RespType>(resp), offsets, rotation_angles, shells, 0.0, tol,
            integrator_test);

    zest::DynamicMDArray<std::array<double, 2>, 2>
    transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zdm::IsotropicZernikeExpansion<double> distribution
        = zest::zt::IsotropicZernikeTransformerNormalGeo<>(dist_order).forward_transform(
                std::forward<DistType>(dist), 1.0, dist_order);

    zdm::SHExpansionVector<double> response{shells.size(), resp_order};
    zdm::ResponseTransformer(resp_order)
        .forward_transform(std::forward<RespType>(resp), shells, response);

    zdm::zebra::TransverseAngleIntegrator<zdm::DistType::iso, zdm::RespType::aniso>(dist_order, resp_order)
        .integrate(distribution, response, offsets, rotation_angles, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success
                && is_close(integrator_test[i, j][0], transformer_test[i, j][0], tol)
                && is_close(integrator_test[i, j][1], transformer_test[i, j][1], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        integrator_test[i, j][0], integrator_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        transformer_test[i, j][0], transformer_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        1.0 - integrator_test[i, j][0]/transformer_test[i, j][0],
                        1.0 - integrator_test[i, j][1]/transformer_test[i, j][1]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

template <typename DistType, typename RespType>
[[nodiscard]] bool test_mutual_convergence_transverse_aniso_aniso(
    DistType&& dist, RespType&& resp, std::span<const zdm::la::Vector<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    constexpr double tol = 1.0e-7;

    zest::DynamicMDArray<std::array<double, 2>, 2>
    integrator_test{offsets.size(), shells.size()};

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(
            std::forward<DistType>(dist), std::forward<RespType>(resp),
            offsets, rotation_angles, shells, 0.0, tol, integrator_test);

    zest::DynamicMDArray<std::array<double, 2>, 2>
    transformer_test{offsets.size(), shells.size()};

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zdm::ZernikeExpansion<double> distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(dist_order).forward_transform(
                std::forward<DistType>(dist), 1.0, dist_order);

    zdm::SHExpansionVector<double> response{shells.size(), resp_order};
    zdm::ResponseTransformer(resp_order)
        .forward_transform(std::forward<RespType>(resp), shells, response);

    zdm::zebra::TransverseAngleIntegrator<zdm::DistType::aniso, zdm::RespType::aniso>(dist_order, resp_order)
        .integrate( distribution, response, offsets, rotation_angles, shells, transformer_test);

    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            success = success
                && is_close(integrator_test[i, j][0], transformer_test[i, j][0], tol)
                && is_close(integrator_test[i, j][1], transformer_test[i, j][1], tol);
        }
    }

    if (!success)
    {
        std::println("integrator");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        integrator_test[i, j][0], integrator_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\ntransformer");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        transformer_test[i, j][0], transformer_test[i, j][1]);
            }
            std::println("");
        }

        std::println("\nrelative error");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::print("[{:.16e}, {:.16e}] ",
                        1.0 - integrator_test[i, j][0]/transformer_test[i, j][0],
                        1.0 - integrator_test[i, j][1]/transformer_test[i, j][1]);
            }
            std::println("");
        }
        std::println("");
    }

    return success;
}

} // namespace

int main()
{
    [[maybe_unused]] auto gaussian_isotropic = [](double speed){
        constexpr double disp = 0.4;
        const double ratio = speed/disp;
        return std::exp(-ratio*ratio);
    };

    [[maybe_unused]] auto gaussian = [](const std::array<double, 3>& v){
        constexpr double disp = 0.4;
        const double speed = zdm::la::length(v);
        const double ratio = speed/disp;
        return std::exp(-ratio*ratio);
    };

    [[maybe_unused]] auto aniso_gaussian = [](const std::array<double, 3>& v)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        return std::exp(-0.5*quadratic_form(sigma, v));
    };

    [[maybe_unused]] auto test_response = [](
        [[maybe_unused]] double shell, [[maybe_unused]]
        double lon, [[maybe_unused]] double colat) -> double
    {
        constexpr double slope = 10.0;
        const std::array<double, 3> dir
            = zdm::coordinates::spherical_to_cartesian_phys(lon, colat);
        return 0.5*(1.0 + std::tanh(slope*(dir[0] + (2.0/1.5)*shell - 1.0)));
    };

    [[maybe_unused]] auto smooth_exponential = [](
        double shell, double longitude, double colatitude)
    {
        static const std::array<double, 3> ref_dir
            = zdm::la::normalize(std::array<double, 3>{0.5, 0.5, 0.5});
        const std::array<double, 3> dir
            = zdm::coordinates::spherical_to_cartesian_phys(longitude, colatitude);
        constexpr double rate = 2.0;
        const double u2 = shell*shell;
        const double u4 = u2*u2;
        return (u4/(1 + u4))*std::exp(rate*(zdm::la::dot(dir, ref_dir)));
    };

   const std::vector<zdm::la::Vector<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0},
        {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0},
        {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0},
        {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}
    };

    const std::vector<double> rotation_angles = {
        0.0*std::numbers::pi, 0.25*std::numbers::pi, 0.5*std::numbers::pi,
        0.75*std::numbers::pi, 1.0*std::numbers::pi, 1.25*std::numbers::pi,
        1.5*std::numbers::pi, 1.75*std::numbers::pi, 0.0*std::numbers::pi,
        0.25*std::numbers::pi, 0.5*std::numbers::pi, 0.75*std::numbers::pi,
        1.0*std::numbers::pi, 1.25*std::numbers::pi, 1.5*std::numbers::pi,
        1.75*std::numbers::pi
    };

    const std::vector<double> shells = {
        0.0, 0.15//, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    assert(test_mutual_convergence_iso_iso(gaussian_isotropic, offsets, shells));
    assert(test_mutual_convergence_transverse_iso_iso(gaussian_isotropic, offsets, shells));

    assert(test_mutual_convergence_aniso_iso(gaussian, offsets, shells));
    assert(test_mutual_convergence_aniso_iso(aniso_gaussian, offsets, shells));

    assert(test_mutual_convergence_transverse_aniso_iso(gaussian, offsets, shells));
    assert(test_mutual_convergence_transverse_aniso_iso(aniso_gaussian, offsets, shells));

    assert(test_mutual_convergence_iso_aniso(gaussian_isotropic, smooth_exponential, offsets, shells, rotation_angles));
    assert(test_mutual_convergence_transverse_iso_aniso(gaussian_isotropic, test_response, offsets, shells, rotation_angles));

    assert(test_mutual_convergence_aniso_aniso(gaussian, smooth_exponential, offsets, shells, rotation_angles));
    assert(test_mutual_convergence_aniso_aniso(aniso_gaussian, smooth_exponential, offsets, shells, rotation_angles));

    assert(test_mutual_convergence_transverse_aniso_aniso(gaussian, test_response, offsets, shells, rotation_angles));
    assert(test_mutual_convergence_transverse_aniso_aniso(aniso_gaussian, test_response, offsets, shells, rotation_angles));
}
