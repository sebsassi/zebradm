#pragma once

#include <vector>
#include <span>
#include <functional>
#include <cmath>

#include "linalg.hpp"
#include "align_z.hpp"

#include "hypercube_integrator.hpp"
#include "array_arithmetic.hpp"

#include "zest/md_span.hpp"

class RadonIntegrator
{
public:
    RadonIntegrator() = default;

    /*
    Parameters:
    `distribution`: velocity distribution. See notes for details.
    `response`: response function. See notes for details.
    `boosts`: negative average velocity of the distribution. See notes for units.
    `min_speeds`: smallest allowed speed. See notes for units.
    `abserr`: maximum absolute error demanded from integration.
    `relerr`: maximum relative error demanded from integration.

    Notes:

    This function expects velocity units such that the distribution is zero for speeds above 1.0.

    `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist>
    void integrate(
        Dist&& distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, double abserr, double relerr, zest::MDSpan<double, 2> out)
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate(
                        distribution, boosts[i], min_speeds[j], abserr, relerr);
        }
    }

    /*
    Parameters:
    `distribution`: velocity distribution. See notes for details.
    `response`: response function. See notes for details.
    `boosts`: negative average velocities of the distribution. See notes for units.
    `min_speeds`: smallest allowed speeds. See notes for units.
    `eras`: Earth rotation angles.
    `abserr`: maximum absolute error demanded from integration.
    `relerr`: maximum relative error demanded from integration.

    Notes:

    This function expects velocity units such that the distribution is zero for speeds above 1.0.

    The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `response` coordinate system to the x-axis of the `distribution` coordinate system.

    `distribution` and `boosts` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    void integrate(
        Dist&& distribution, Resp&& response, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, std::span<const double> eras, double abserr, double relerr, zest::MDSpan<double, 2> out)
    {
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
                out(i, j) = integrate(
                        distribution, response, boosts[i], min_speeds[j],
                        eras[i], abserr, relerr);
        }
    }
    
    /*
    Parameters:
    `distribution`: velocity distribution. See notes for details.
    `response`: response function. See notes for details.
    `boost`: negative average velocity of the distribution. See notes for units.
    `min_speed`: smallest allowed speed. See notes for units.
    `abserr`: maximum absolute error demanded from integration.
    `relerr`: maximum relative error demanded from integration.

    Notes:

    This function expects velocity units such that the distribution is zero for speeds above 1.0.

    `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist>
    [[nodiscard]] double integrate(
        Dist&& distribution, const Vector<double, 3>& boost, double min_speed, double abserr, double relerr)
    {
        const double boost_speed = length(boost);
        if (min_speed > 1.0 + boost_speed) return 0.0;
        const Matrix<double, 3, 3> align_z_transp
            = rotation_matrix_align_z_transp(normalize(boost));
        const double boost_speed_sq = boost_speed*boost_speed;
        auto integrand = [&](const Vector<double, 3>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double t = coords[2];
            const double zmax = std::min(1.0, (1.0 - v*v - boost_speed_sq)/(2.0*v*boost_speed));
            const double z = 0.5*((1.0 + zmax)*t - (1.0 - zmax));
            const double v_perp = v*std::sqrt((1.0 - z)*(1.0 + z));

            // `velocity` is in coordinates with z-axis in direction of `boost`.
            const Vector<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), v*z
            };

            // `align_z_transp` rotates `velocity` back into the original coordinates.
            return 0.5*(1 + zmax)*v*distribution(matmul(align_z_transp, velocity) + boost);
        };

        Integrator3D::Limits limits = {
            {min_speed, 0.0, -1.0},
            {1.0 + boost_speed, 2.0*std::numbers::pi, 1.0}
        };
        return (2.0*std::numbers::pi)*isotropic_integrator.integrate(
                integrand, limits, abserr, relerr).val;
    }
    
    /*
    Parameters:
    `distribution`: velocity distribution. See notes for details.
    `response`: response function. See notes for details.
    `boost`: negative average velocity of the distribution. See notes for units.
    `min_speed`: smallest allowed speed. See notes for units.
    `era`: Earth rotation angle.
    `abserr`: maximum absolute error demanded from integration.
    `relerr`: maximum relative error demanded from integration.

    Notes:

    This function expects velocity units such that the distribution is zero for speeds above 1.0.

    The function `response` has parameters min_speed, azimuth, and colatitude. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `response` coordinate system to the x-axis of the `distribution` coordinate system.

    `distribution` and `boost` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    [[nodiscard]] double integrate(
        Dist&& distribution, Resp&& response, const Vector<double, 3>& boost, double min_speed, double era, double abserr, double relerr)
    {
        const double boost_speed = length(boost);
        if (min_speed - boost_speed > 1.0) return 0.0;

        // `boost_era` is in the same coordinates as `response`.
        const Vector<double, 3> boost_era = {
            std::cos(era)*boost[0] - std::sin(era)*boost[1],
            std::sin(era)*boost[0] + std::cos(era)*boost[1],
            boost[2]
        };
        const Matrix<double, 3, 3> align_z_transp_resp
            = rotation_matrix_align_z_transp(normalize(boost_era));
        const Matrix<double, 3, 3> align_z_transp_dist
            = rotation_matrix_align_z_transp(normalize(boost));
        auto integrand = [&](const Vector<double, 2>& coords)
        {
            const double azimuth = coords[0];
            const double z = coords[1];
            const double perp = std::sqrt((1.0 - z)*(1.0 + z));

            // `recoil_dir` is in coordinates with z-axis in direction of `boost`.
            const Vector<double, 3> recoil_dir = {
                perp*std::cos(azimuth), perp*std::sin(azimuth), z
            };

            // `recoil_dir_resp` is in the same coordinates as `response`
            const Vector<double, 3> recoil_dir_resp
                = matmul(align_z_transp_resp, recoil_dir);
            
            // `recoil_dir_dist` is in the same coordinates as `distribution`
            const Vector<double, 3> recoil_dir_dist
                = matmul(align_z_transp_dist, recoil_dir);
            
            const double colat_resp = std::acos(recoil_dir_resp[2]);
            const double azimuth_resp
                = std::atan2(recoil_dir_resp[1], recoil_dir_resp[0]);
            const double resp = response(min_speed, azimuth_resp, colat_resp);

            return resp*velocity_integral(
                    distribution, boost, min_speed, recoil_dir_dist, abserr, relerr);
        };

        const double zmin = std::max(-(1.0 + min_speed)/boost_speed, -1.0);
        const double zmax = std::min((1.0 - min_speed)/boost_speed, 1.0);
        Integrator2D::Limits limits = {
            {0.0, zmin}, {2.0*std::numbers::pi, zmax}
        };
        return angle_integrator.integrate(
                integrand, limits, abserr, relerr).val;
    }

private:
    template <typename Dist>
    double velocity_integral(
        Dist&& distribution, const Vector<double, 3>& boost_dist,
        double min_speed, const Vector<double, 3>& recoil_dir_dist,
        double abserr, double relerr)
    {
        const double radon_parameter
            = min_speed + dot(boost_dist, recoil_dir_dist);
        const double w = std::fabs(radon_parameter);
        if (w > 1.0) return 0.0;

        const Matrix<double, 3, 3> align_z_transp
            = rotation_matrix_align_z_transp(recoil_dir_dist);
        auto integrand = [&](const Vector<double, 2>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double v_perp = std::sqrt((v - radon_parameter)*(v + radon_parameter));


            // `velocity` is in coordinates with z-axis in direction of `recoil_dir`.
            const Vector<double, 3> velocity = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), radon_parameter
            };

            // `align_z_transp` rotates `velocity` back into the original coordinates.
            return v*distribution(matmul(align_z_transp, velocity));
        };

        Integrator2D::Limits limits = {
            {w, 0.0}, {1.0, 2.0*std::numbers::pi}
        };
        return velocity_integrator.integrate(
                integrand, limits, abserr, relerr).val;
    }

    using Integrator2D = cubage::HypercubeIntegrator<std::array<double, 2>, double>;
    using Integrator3D = cubage::HypercubeIntegrator<std::array<double, 3>, double>;
    Integrator3D isotropic_integrator;
    Integrator2D velocity_integrator;
    Integrator2D angle_integrator;
};