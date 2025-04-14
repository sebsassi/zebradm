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

#include "cubage/array_arithmetic.hpp"
#include "cubage/hypercube_integrator.hpp"

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
        @brief Angle integrated Radon transform of a disitribution on an offset unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution distribution function
        @param offsets offsets of the distribution
        @param shells distances of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    template <typename Dist>
    void integrate(
        Dist&& distribution, std::span<const std::array<double, 3>> offsets,
        std::span<const double> shells, double abserr, double relerr,
        zest::MDSpan<double, 2> out,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
                out(i, j) = integrate(
                        distribution, offsets[i], shells[j], abserr, relerr, 
                        max_subdiv);
        }
    }

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a
        velocity disitribution on an offset unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution distribution function
        @param offsets offsets of the distribution
        @param shells distances of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    template <typename Dist>
    void integrate_transverse(
        Dist&& distribution, std::span<const std::array<double, 3>> offsets,
        std::span<const double> shells, double abserr, double relerr,
        zest::MDSpan<std::array<double, 2>, 2> out,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
                out(i, j) = integrate_transverse(
                        distribution, offsets[i], shells[j], abserr, relerr, 
                        max_subdiv);
        }
    }

    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution distribution function
        @param response response function
        @param offsets offsets of the distribution
        @param rotation_angles z-axis rotation angles between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    void integrate(
        Dist&& distribution, Resp&& response,
        std::span<const std::array<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        double abserr, double relerr, zest::MDSpan<double, 2> out,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
                out(i, j) = integrate(
                        distribution, response, offsets[i], rotation_angles[i],
                        shells[j], abserr, relerr, max_subdiv);
        }
    }

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity
        disitribution on an offset unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution distribution function
        @param response response function
        @param offsets offsets of the distribution
        @param rotation_angles z-axis rotation angles between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`
        
        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    void integrate_transverse(
        Dist&& distribution, Resp&& response,
        std::span<const std::array<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        double abserr, double relerr, zest::MDSpan<std::array<double, 2>, 2> out,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
                out(i, j) = integrate_transverse(
                        distribution, response, offsets[i], rotation_angles[i],
                        shells[j], abserr, relerr, max_subdiv);
        }
    }
    
    /**
        @brief Angle integrated Radon transform of a disitribution on an offset unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution distribution function
        @param offset offset of the distribution
        @param shell distance of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shell` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the `offset`. 

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    template <typename Dist>
    [[nodiscard]] double integrate(
        Dist&& distribution, const std::array<double, 3>& offset, double shell,
        double abserr, double relerr,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double offset_len = length(offset);
        if (shell > 1.0 + offset_len) return 0.0;
        const Matrix<double, 3, 3> align_z_transp
            = detail::rotation_matrix_align_z_transp(normalize(offset));
        const double offset_len_sq = offset_len*offset_len;
        auto integrand = [&](const std::array<double, 3>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double t = coords[2];
            const double zmax = std::min(1.0, (1.0 - v*v - offset_len_sq)/(2.0*v*offset_len));
            const double z = 0.5*((1.0 + zmax)*t - (1.0 - zmax));
            const double v_perp = v*std::sqrt((1.0 - z)*(1.0 + z));

            // `point` is in coordinates with z-axis in direction of `offset`.
            const std::array<double, 3> point = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), v*z
            };

            // `align_z_transp` rotates `point` back into the original coordinates.
            return 0.5*(1 + zmax)*v*distribution(matmul(align_z_transp, point) + offset);
        };

        Integrator3D<double>::Limits limits = {
            {shell, 0.0, -1.0},
            {1.0 + offset_len, 2.0*std::numbers::pi, 1.0}
        };
        return (2.0*std::numbers::pi)*isotropic_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a
        velocity disitribution on an offset unit ball.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`

        @param distribution distribution function
        @param offset offset of the distribution
        @param shell distance of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shell` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the `offset`.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    template <typename Dist>
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        Dist&& distribution, const std::array<double, 3>& offset, double shell,
        double abserr, double relerr,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double offset_len = length(offset);
        if (shell > 1.0 + offset_len) return {};
        const Matrix<double, 3, 3> align_z_transp
            = detail::rotation_matrix_align_z_transp(normalize(offset));
        const double offset_len_sq = offset_len*offset_len;
        const double shell_sq = shell*shell;
        auto integrand = [&](const std::array<double, 3>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double t = coords[2];
            const double zmax = std::min(1.0, (1.0 - v*v - offset_len_sq)/(2.0*v*offset_len));
            const double z = 0.5*((1.0 + zmax)*t - (1.0 - zmax));
            const double v_perp = v*std::sqrt((1.0 - z)*(1.0 + z));

            // `point` is in coordinates with z-axis in direction of `offset`.
            const std::array<double, 3> point = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), v*z
            };

            // `align_z_transp` rotates `velocity` back into the original coordinates.
            const double integrand = 0.5*(1 + zmax)*v*distribution(matmul(align_z_transp, point) + offset);
            return std::array<double, 2>{
                integrand, (v*v - shell_sq)*integrand
            };
        };

        Integrator3D<std::array<double, 2>>::Limits limits = {
            {shell, 0.0, -1.0},
            {1.0 + offset_len, 2.0*std::numbers::pi, 1.0}
        };
        return (2.0*std::numbers::pi)*transverse_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution distribution function
        @param response response function
        @param offset offset of the distribution
        @param rotation_angle z-axis rotation angle between distribution and response coordinates
        @param shell distance of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    [[nodiscard]] double integrate(
        Dist&& distribution, Resp&& response,
        const std::array<double, 3>& offset, double rotation_angle,
        double shell, double abserr, double relerr,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double offset_len = length(offset);
        if (shell - offset_len > 1.0) return 0.0;

        // Rotation from `distribution` coordinates to `response` coordinates.
        const Matrix<double, 3, 3> dist_to_resp = {
            std::array<double, 3>{std::cos(rotation_angle), std::sin(rotation_angle), 0.0},
            std::array<double, 3>{-std::sin(rotation_angle), std::cos(rotation_angle), 0.0},
            std::array<double, 3>{0.0, 0.0, 1.0}
        };

        // Rotation from coordinates where the z-axis is in the direction of `offset` to `distribution` coordinates.
        const Matrix<double, 3, 3> offset_to_dist
            = detail::rotation_matrix_align_z_transp(normalize(offset));
        

        // Rotation from coordinates where the z-axis is in the direction of `offset` to `response` coordinates.
        const Matrix<double, 3, 3> offset_to_resp
            = matmul(dist_to_resp, offset_to_dist);

        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double azimuth = coords[0];
            const double z = coords[1];
            const double perp = std::sqrt((1.0 - z)*(1.0 + z));

            // `normal` is in coordinates with z-axis in direction of `offset`.
            const std::array<double, 3> normal = {
                perp*std::cos(azimuth), perp*std::sin(azimuth), z 
            };

            // `normal` is in the same coordinates as `response`
            const std::array<double, 3> normal_resp
                = matmul(offset_to_resp, normal);
            
            // `normal_dist` is in the same coordinates as `distribution`
            const std::array<double, 3> normal_dist
                = matmul(offset_to_dist, normal);
            
            const auto& [resp_az, resp_colat, resp_mag]
                = coordinates::cartesian_to_spherical_phys(normal_resp);
            const double resp = response(shell, resp_az, resp_colat);

            return resp*radon_integral(
                    distribution, offset, shell, normal_dist, abserr, relerr, max_subdiv);
        };

        const double zmin = std::max(-(1.0 + shell)/offset_len, -1.0);
        const double zmax = std::min((1.0 - shell)/offset_len, 1.0);
        Integrator2D<double>::Limits limits = {
            {0.0, zmin}, {2.0*std::numbers::pi, zmax}
        };
        return angle_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }
    
    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a velocity
        disitribution on an offset unit ball, combined with an angle-dependent response.

        @tparam Dist callable accepting a `std::array<double, 3>` and returning `double`
        @tparam Resp callable accepting three `double`s and returning `double`

        @param distribution distribution function
        @param response response function
        @param offset offset of the distribution
        @param rotation_angle z-axis rotation angle between distribution and response coordinates
        @param shell distance of integration planes from the origin
        @param abserr desired absolute error passed to integrator
        @param relerr desired relative error passed to integrator
        
        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    template <typename Dist, typename Resp>
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        Dist&& distribution, Resp&& response,
        const std::array<double, 3>& offset, double rotation_angle,
        double shell, double abserr, double relerr,
        std::size_t max_subdiv = std::numeric_limits<std::size_t>::max())
    {
        const double offset_len = length(offset);
        if (shell - offset_len > 1.0) return {};

        // Rotation from `distribution` coordinates to `response` coordinates.
        const Matrix<double, 3, 3> dist_to_resp = {
            std::array<double, 3>{std::cos(rotation_angle), std::sin(rotation_angle), 0.0},
            std::array<double, 3>{-std::sin(rotation_angle), std::cos(rotation_angle), 0.0},
            std::array<double, 3>{0.0, 0.0, 1.0}
        };

        // Rotation from coordinates where the z-axis is in the direction of `offset` to `distribution` coordinates.
        const Matrix<double, 3, 3> offset_to_dist
            = detail::rotation_matrix_align_z_transp(normalize(offset));
        

        // Rotation from coordinates where the z-axis is in the direction of `offset` to `response` coordinates.
        const Matrix<double, 3, 3> offset_to_resp
            = matmul(dist_to_resp, offset_to_dist);

        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double azimuth = coords[0];
            const double z = coords[1];
            const double perp = std::sqrt((1.0 - z)*(1.0 + z));

            // `normal` is in coordinates with z-axis in direction of `offset`.
            const std::array<double, 3> normal = {
                perp*std::cos(azimuth), perp*std::sin(azimuth), z 
            };

            // `normal_resp` is in the same coordinates as `response`
            const std::array<double, 3> normal_resp
                = matmul(offset_to_resp, normal);
            
            // `normal_dist` is in the same coordinates as `distribution`
            const std::array<double, 3> normal_dist
                = matmul(normal_to_dist, normal);
            
            const auto& [resp_az, resp_colat, resp_mag]
                = coordinates::cartesian_to_spherical_phys(normal_resp);
            const double resp = response(shell, resp_az, resp_colat);

            const std::array<double, 2> integral = transverse_radon_integral(
                    distribution, offset, shell, normal_dist, abserr, relerr, max_subdiv);
            return std::array<double, 2>{resp*integral[0], resp*integral[1]};
        };

        const double zmin = std::max(-(1.0 + shell)/offset_len, -1.0);
        const double zmax = std::min((1.0 - shell)/offset_len, 1.0);
        Integrator2D<std::array<double, 2>>::Limits limits = {
            {0.0, zmin}, {2.0*std::numbers::pi, zmax}
        };
        return transverse_angle_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

private:
    template <typename Dist>
    double radon_integral(
        Dist&& distribution, const std::array<double, 3>& offset_dist,
        double shell, const std::array<double, 3>& normal_dist,
        double abserr, double relerr, std::size_t max_subdiv)
    {
        const double radon_parameter
            = shell + dot(offset_dist, normal_dist);
        const double w = std::fabs(radon_parameter);
        if (w > 1.0) return 0.0;

        const Matrix<double, 3, 3> to_dist_coords
            = detail::rotation_matrix_align_z_transp(normal_dist);
        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double v_perp = std::sqrt((v - radon_parameter)*(v + radon_parameter));

            // `point` is in coordinates with z-axis in direction of `normal`.
            const std::array<double, 3> point = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), radon_parameter
            };

            // `align_z_transp` rotates `point` back into distribution coordinates
            return v*distribution(matmul(to_dist_coords, point));
        };

        Integrator2D<double>::Limits limits = {
            {w, 0.0}, {1.0, 2.0*std::numbers::pi}
        };
        return radon_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

    template <typename Dist>
    std::array<double, 2> transverse_radon_integral(
        Dist&& distribution, const std::array<double, 3>& offset_dist,
        double shell, const std::array<double, 3>& normal_dist,
        double abserr, double relerr, std::size_t max_subdiv)
    {
        const double radon_parameter
            = shell + dot(offset_dist, normal_dist);
        const double w = std::fabs(radon_parameter);
        if (w > 1.0) return {};

        const Matrix<double, 3, 3> to_dist_coords
            = detail::rotation_matrix_align_z_transp(normal_dist);
        auto integrand = [&](const std::array<double, 2>& coords)
        {
            const double v = coords[0];
            const double azimuth = coords[1];
            const double v_perp
                = std::sqrt((v - radon_parameter)*(v + radon_parameter));

            // `point` is in coordinates with z-axis in direction of `normal`.
            const std::array<double, 3> point = {
                v_perp*std::cos(azimuth), v_perp*std::sin(azimuth), radon_parameter
            };
            const std::array<double, 3> point_dist
                = matmul(to_dist_coords, point);
            const std::array<double, 3> point_offset
                = point_dist - offset_dist;

            const double p_offset_sq = dot(point_offset, point_offset);
            const double p_offset_perp_sq = p_offset_sq - shell*shell;

            // `align_z_transp` rotates `point` back into distribution coordinates
            const double dist = distribution(point_dist);
            return std::array<double, 2>{v*dist, v*p_offset_perp_sq*dist};
        };

        Integrator2D<std::array<double, 2>>::Limits limits = {
            {w, 0.0}, {1.0, 2.0*std::numbers::pi}
        };
        return transverse_radon_integrator.integrate(
                integrand, limits, abserr, relerr, max_subdiv).value.val;
    }

    template <typename T>
    using Integrator2D = cubage::HypercubeIntegrator<std::array<double, 2>, T>;
    template <typename T>
    using Integrator3D = cubage::HypercubeIntegrator<std::array<double, 3>, T>;
    template <typename T>
    using Integrator4D = cubage::HypercubeIntegrator<std::array<double, 4>, T>;

    Integrator3D<double> isotropic_integrator;
    Integrator2D<double> radon_integrator;
    Integrator2D<double> angle_integrator;
    Integrator4D<double> anisotropic_integrator;
    Integrator3D<std::array<double, 2>> transverse_integrator;
    Integrator2D<std::array<double, 2>> transverse_radon_integrator;
    Integrator2D<std::array<double, 2>> transverse_angle_integrator;
};

} // namspace integrate
} // namespace zdm
