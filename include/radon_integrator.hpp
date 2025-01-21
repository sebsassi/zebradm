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
#pragma once

#include <vector>
#include <span>
#include <functional>
#include <cmath>
#include <limits>
#include <array>

#include <zest/md_span.hpp>

#include "linalg.hpp"
#include "align_z.hpp"

#include "array_arithmetic.hpp"
#include "hypercube_integrator.hpp"

#include "coordinate_transforms.hpp"

namespace zdm
{
namespace integrate
{

class RadonAngleIntegrator
{
public:
    RadonAngleIntegrator() = default;

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution velocity distribution; see notes for details
        @param boosts negative average velocity of the distribution; see notes for units
        @param min_speeds smallest allowed speed. See notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist>
    void integrate(
        Dist&& distribution, std::span<const std::array<double, 3>> boosts, std::span<const double> min_speeds, double abserr, double relerr, zest::MDSpan<double, 2> out, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate(
                        distribution, boosts[i], min_speeds[j], abserr, relerr, 
                        max_subdiv);
        }
    }

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity disitribution on a boosted unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution velocity distribution; see notes for details
        @param boosts negative average velocity of the distribution; see notes for units
        @param min_speeds smallest allowed speed. See notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist>
    void integrate_transverse(
        Dist&& distribution, std::span<const std::array<double, 3>> boosts, std::span<const double> min_speeds, double abserr, double relerr, zest::MDSpan<std::array<double, 2>, 2> out, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate_transverse(
                        distribution, boosts[i], min_speeds[j], abserr, relerr, 
                        max_subdiv);
        }
    }

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution velocity distribution; see notes for details
        @param response response function; see notes for details
        @param boosts negative average velocities of the distribution; see notes for units
        @param eras Earth rotation angles
        @param min_speeds smallest allowed speeds; ee notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `response` coordinate system to the x-axis of the `distribution` coordinate system.

        @note `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    void integrate(
        Dist&& distribution, Resp&& response, std::span<const std::array<double, 3>> boosts, std::span<const double> eras, std::span<const double> min_speeds, double abserr, double relerr, zest::MDSpan<double, 2> out, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate(
                        distribution, response, boosts[i], eras[i],
                        min_speeds[j], abserr, relerr, max_subdiv);
        }
    }

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution velocity distribution; see notes for details
        @param response response function; see notes for details
        @param boosts negative average velocities of the distribution; see notes for units
        @param eras Earth rotation angles
        @param min_speeds smallest allowed speeds; ee notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `response` coordinate system to the x-axis of the `distribution` coordinate system.

        @note `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    void integrate_transverse(
        Dist&& distribution, Resp&& response, std::span<const std::array<double, 3>> boosts, std::span<const double> eras, std::span<const double> min_speeds, double abserr, double relerr, zest::MDSpan<std::array<double, 2>, 2> out, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate_transverse(
                        distribution, response, boosts[i], eras[i],
                        min_speeds[j], abserr, relerr, max_subdiv);
        }
    }
    
    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution velocity distribution; see notes for details
        @param boost negative average velocity of the distribution; see notes for units
        @param min_speed smallest allowed speed; see notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist>
    [[nodiscard]] double integrate(
        Dist&& distribution, const std::array<double, 3>& boost, double min_speed, double abserr, double relerr, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double boost_speed = length(boost);
        if (min_speed > 1.0 + boost_speed) return 0.0;
        const Matrix<double, 3, 3> align_z_transp
            = detail::rotation_matrix_align_z_transp(normalize(boost));
        const double boost_speed_sq = boost_speed*boost_speed;
        auto integrand = [&](const std::array<double, 3>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double t = coords[2];
            const double zmax = std::min(1.0, (1.0 - v*v - boost_speed_sq)/(2.0*v*boost_speed));
            const double z = 0.5*((1.0 + zmax)*t - (1.0 - zmax));
            const double v_perp = v*std::sqrt((1.0 - z)*(1.0 + z));

            // `velocity` is in coordinates with z-axis in direction of `boost`.
            const std::array<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), v*z
            };

            // `align_z_transp` rotates `velocity` back into the original coordinates.
            return 0.5*(1 + zmax)*v*distribution(matmul(align_z_transp, velocity) + boost);
        };

        Integrator3D<double>::Limits limits = {
            {min_speed, 0.0, -1.0},
            {1.0 + boost_speed, 2.0*std::numbers::pi, 1.0}
        };
        return (2.0*std::numbers::pi)*isotropic_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity disitribution on a boosted unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution velocity distribution; see notes for details
        @param boost negative average velocity of the distribution; see notes for units
        @param min_speed smallest allowed speed; see notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist>
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        Dist&& distribution, const std::array<double, 3>& boost, double min_speed, double abserr, double relerr, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double boost_speed = length(boost);
        if (min_speed > 1.0 + boost_speed) return {};
        const Matrix<double, 3, 3> align_z_transp
            = detail::rotation_matrix_align_z_transp(normalize(boost));
        const double boost_speed_sq = boost_speed*boost_speed;
        const double min_speed_sq = min_speed*min_speed;
        auto integrand = [&](const std::array<double, 3>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double t = coords[2];
            const double zmax = std::min(1.0, (1.0 - v*v - boost_speed_sq)/(2.0*v*boost_speed));
            const double z = 0.5*((1.0 + zmax)*t - (1.0 - zmax));
            const double v_perp = v*std::sqrt((1.0 - z)*(1.0 + z));

            // `velocity` is in coordinates with z-axis in direction of `boost`.
            const std::array<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), v*z
            };

            // `align_z_transp` rotates `velocity` back into the original coordinates.
            const double integrand = 0.5*(1 + zmax)*v*distribution(matmul(align_z_transp, velocity) + boost);
            return std::array<double, 2>{
                integrand, (v*v - min_speed_sq)*integrand
            };
        };

        Integrator3D<std::array<double, 2>>::Limits limits = {
            {min_speed, 0.0, -1.0},
            {1.0 + boost_speed, 2.0*std::numbers::pi, 1.0}
        };
        return (2.0*std::numbers::pi)*transverse_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution velocity distribution; see notes for details
        @param response response function; see notes for details
        @param boost negative average velocity of the distribution; see notes for units
        @param era Earth rotation angle
        @param min_speed smallest allowed speed; see notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    [[nodiscard]] double integrate(
        Dist&& distribution, Resp&& response, const std::array<double, 3>& boost, double era, double min_speed, double abserr, double relerr, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double boost_speed = length(boost);
        if (min_speed - boost_speed > 1.0) return 0.0;

        // Rotation from `distribution` coordinates to `response` coordinates.
        const Matrix<double, 3, 3> dist_to_resp = {
            std::array<double, 3>{std::cos(era), std::sin(era), 0.0},
            std::array<double, 3>{-std::sin(era), std::cos(era), 0.0},
            std::array<double, 3>{0.0, 0.0, 1.0}
        };

        // Rotation from coordinates where the z-axis is in the direction of `boost` to `distribution` coordinates.
        const Matrix<double, 3, 3> boost_to_dist
            = detail::rotation_matrix_align_z_transp(normalize(boost));
        

        // Rotation from coordinates where the z-axis is in the direction of `boost` to `response` coordinates.
        const Matrix<double, 3, 3> boost_to_resp
            = matmul(dist_to_resp, boost_to_dist);

        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double azimuth = coords[0];
            const double z = coords[1];
            const double perp = std::sqrt((1.0 - z)*(1.0 + z));

            // `recoil_dir` is in coordinates with z-axis in direction of `boost`.
            const std::array<double, 3> recoil_dir = {
                perp*std::cos(azimuth), perp*std::sin(azimuth), z 
            };

            // `recoil_dir_resp` is in the same coordinates as `response`
            const std::array<double, 3> recoil_dir_resp
                = matmul(boost_to_resp, recoil_dir);
            
            // `recoil_dir_dist` is in the same coordinates as `distribution`
            const std::array<double, 3> recoil_dir_dist
                = matmul(boost_to_dist, recoil_dir);
            
            const auto& [resp_az, resp_colat, resp_mag]
                = coordinates::cartesian_to_spherical_phys(recoil_dir_resp);
            const double resp = response(min_speed, resp_az, resp_colat);

            return resp*velocity_integral(
                    distribution, boost, min_speed, recoil_dir_dist, abserr, relerr, max_subdiv);
        };

        const double zmin = std::max(-(1.0 + min_speed)/boost_speed, -1.0);
        const double zmax = std::min((1.0 - min_speed)/boost_speed, 1.0);
        Integrator2D<double>::Limits limits = {
            {0.0, zmin}, {2.0*std::numbers::pi, zmax}
        };
        return angle_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution velocity distribution; see notes for details
        @param response response function; see notes for details
        @param boost negative average velocity of the distribution; see notes for units
        @param era Earth rotation angle
        @param min_speed smallest allowed speed; see notes for units
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        @note This function expects velocity units such that the distribution is zero for speeds above 1.0.

        @note The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        Dist&& distribution, Resp&& response, const std::array<double, 3>& boost, double era, double min_speed, double abserr, double relerr, std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double boost_speed = length(boost);
        if (min_speed - boost_speed > 1.0) return {};

        // Rotation from `distribution` coordinates to `response` coordinates.
        const Matrix<double, 3, 3> dist_to_resp = {
            std::array<double, 3>{std::cos(era), std::sin(era), 0.0},
            std::array<double, 3>{-std::sin(era), std::cos(era), 0.0},
            std::array<double, 3>{0.0, 0.0, 1.0}
        };

        // Rotation from coordinates where the z-axis is in the direction of `boost` to `distribution` coordinates.
        const Matrix<double, 3, 3> boost_to_dist
            = detail::rotation_matrix_align_z_transp(normalize(boost));
        

        // Rotation from coordinates where the z-axis is in the direction of `boost` to `response` coordinates.
        const Matrix<double, 3, 3> boost_to_resp
            = matmul(dist_to_resp, boost_to_dist);

        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double azimuth = coords[0];
            const double z = coords[1];
            const double perp = std::sqrt((1.0 - z)*(1.0 + z));

            // `recoil_dir` is in coordinates with z-axis in direction of `boost`.
            const std::array<double, 3> recoil_dir = {
                perp*std::cos(azimuth), perp*std::sin(azimuth), z 
            };

            // `recoil_dir_resp` is in the same coordinates as `response`
            const std::array<double, 3> recoil_dir_resp
                = matmul(boost_to_resp, recoil_dir);
            
            // `recoil_dir_dist` is in the same coordinates as `distribution`
            const std::array<double, 3> recoil_dir_dist
                = matmul(boost_to_dist, recoil_dir);
            
            const auto& [resp_az, resp_colat, resp_mag]
                = coordinates::cartesian_to_spherical_phys(recoil_dir_resp);
            const double resp = response(min_speed, resp_az, resp_colat);

            const std::array<double, 2> integral = transverse_velocity_integral(
                    distribution, boost, min_speed, recoil_dir_dist, abserr, relerr, max_subdiv);
            return std::array<double, 2>{resp*integral[0], resp*integral[1]};
        };

        const double zmin = std::max(-(1.0 + min_speed)/boost_speed, -1.0);
        const double zmax = std::min((1.0 - min_speed)/boost_speed, 1.0);
        Integrator2D<std::array<double, 2>>::Limits limits = {
            {0.0, zmin}, {2.0*std::numbers::pi, zmax}
        };
        return transverse_angle_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

private:
    template <typename Dist>
    double velocity_integral(
        Dist&& distribution, const std::array<double, 3>& boost_dist,
        double min_speed, const std::array<double, 3>& recoil_dir_dist,
        double abserr, double relerr, std::size_t max_subdiv)
    {
        const double radon_parameter
            = min_speed + dot(boost_dist, recoil_dir_dist);
        const double w = std::fabs(radon_parameter);
        if (w > 1.0) return 0.0;

        const Matrix<double, 3, 3> to_dist_coords
            = detail::rotation_matrix_align_z_transp(recoil_dir_dist);
        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double v_perp = std::sqrt((v - radon_parameter)*(v + radon_parameter));

            // `velocity` is in coordinates with z-axis in direction of `recoil_dir`.
            const std::array<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), radon_parameter
            };

            // `align_z_transp` rotates `velocity` back into distribution coordinates
            return v*distribution(matmul(to_dist_coords, velocity));
        };

        Integrator2D<double>::Limits limits = {
            {w, 0.0}, {1.0, 2.0*std::numbers::pi}
        };
        return velocity_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

    template <typename Dist>
    std::array<double, 2> transverse_velocity_integral(
        Dist&& distribution, const std::array<double, 3>& boost_dist,
        double min_speed, const std::array<double, 3>& recoil_dir_dist,
        double abserr, double relerr, std::size_t max_subdiv)
    {
        const double radon_parameter
            = min_speed + dot(boost_dist, recoil_dir_dist);
        const double w = std::fabs(radon_parameter);
        if (w > 1.0) return {};

        const Matrix<double, 3, 3> to_dist_coords
            = detail::rotation_matrix_align_z_transp(recoil_dir_dist);
        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double v_perp
                = std::sqrt((v - radon_parameter)*(v + radon_parameter));

            // `velocity` is in coordinates with z-axis in direction of `recoil_dir`.
            const std::array<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), radon_parameter
            };
            const std::array<double, 3> velocity_dist
                = matmul(to_dist_coords, velocity);
            const std::array<double, 3> velocity_boosted
                = velocity_dist - boost_dist;

            const double v_boost_sq = dot(velocity_boosted, velocity_boosted);
            const double v_boost_perp_sq = v_boost_sq - min_speed*min_speed;

            // `align_z_transp` rotates `velocity` back into distribution coordinates
            const double dist = distribution(velocity_dist);
            return std::array<double, 2>{v*dist, v*v_boost_perp_sq*dist};
        };

        Integrator2D<std::array<double, 2>>::Limits limits = {
            {w, 0.0}, {1.0, 2.0*std::numbers::pi}
        };
        return transverse_velocity_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

    template <typename T>
    using Integrator2D = cubage::HypercubeIntegrator<std::array<double, 2>, T>;
    template <typename T>
    using Integrator3D = cubage::HypercubeIntegrator<std::array<double, 3>, T>;
    template <typename T>
    using Integrator4D = cubage::HypercubeIntegrator<std::array<double, 4>, T>;

    Integrator3D<double> isotropic_integrator;
    Integrator2D<double> velocity_integrator;
    Integrator2D<double> angle_integrator;
    Integrator4D<double> anisotropic_integrator;
    Integrator3D<std::array<double, 2>> transverse_integrator;
    Integrator2D<std::array<double, 2>> transverse_velocity_integrator;
    Integrator2D<std::array<double, 2>> transverse_angle_integrator;
};

} // namspace integrate
} // namespace zdm
