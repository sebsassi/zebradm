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

#include <array>
#include <vector>
#include <span>
#include <limits>

#include <zest/md_span.hpp>
#include <zest/zernike_expansion.hpp>
#include <zest/zernike_glq_transformer.hpp>
#include <zest/sh_glq_transformer.hpp>
#include <zest/rotor.hpp>

#include "linalg.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"
#include "zernike_recursions.hpp"
#include "zebra_util.hpp"

namespace zdm
{
namespace zebra
{

/**
    @brief Angle integrated Radon transforms using the Zernike based Radon transform.
*/
class IsotropicAngleIntegrator
{
public:
    IsotropicAngleIntegrator() = default;
    explicit IsotropicAngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boosts velocities of the observer frame at different times.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, std::span<const std::array<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out);
    
    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, const std::array<double, 3>& boost, std::span<const double> min_speeds, std::span<double> out);
    
private:
    void integrate(
        const std::array<double, 3>& boost, std::span<const double> min_speeds,
        std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp;
    zest::zt::RealZernikeExpansionNormalGeo m_rotated_geg_zernike_exp;
    detail::IsotropicAngleIntegratorCore m_integrator_core;
    std::size_t m_dist_order;
};

/**
    @brief Angle integrated Radon transforms with anisotropic response function using the Zernike based Radon transform.
*/
class AnisotropicAngleIntegrator
{
public:
    AnisotropicAngleIntegrator() = default;
    AnisotropicAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());
    
    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    [[nodiscard]] std::size_t
    truncation_order() const noexcept { return m_trunc_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param boosts velocities of the observer frame at different times.
        @param era Earth rotation angles.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, SHExpansionVectorSpan<const std::array<double, 2>> response, std::span<const std::array<double, 3>> boosts, std::span<const double> era, std::span<const double> min_speeds, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param boost velocity of the observer frame.
        @param era Earth rotation angles.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, SHExpansionVectorSpan<const std::array<double, 2>> response, const std::array<double, 3>& boost, double era, std::span<const double> min_speeds, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        SHExpansionVectorSpan<const std::array<double, 2>> response, const std::array<double, 3>& boost, double era, std::span<const double> min_speeds, std::size_t geg_order, std::size_t top_order, std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp;
    std::vector<std::array<double, 2>> m_rotated_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_grids;
    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    detail::AnisotropicAngleIntegratorCore m_integrator_core;
    std::size_t m_dist_order;
    std::size_t m_resp_order;
    std::size_t m_trunc_order;
};

/**
    @brief Angle integrated regular and transverse Radon transforms and using the Zernike based Radon transform.
*/
class IsotropicTransverseAngleIntegrator
{
public:
    IsotropicTransverseAngleIntegrator() = default;
    explicit IsotropicTransverseAngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated nontransverse and transverse Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boosts velocities of the observer frame at different times.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, std::span<const std::array<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<std::array<double, 2>, 2> out);
    
    /**
        @brief Angle integrated nontransverse and transverse Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, const std::array<double, 3>& boost, std::span<const double> min_speeds, std::span<std::array<double, 2>> out);
    
private:
    void integrate(
        const std::array<double, 3>& boost, std::span<const double> min_speeds,
        std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_x;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_y;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_z;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_r2;
    zest::zt::RealZernikeExpansionNormalGeo m_rotated_geg_zernike_exp;
    zest::zt::RealZernikeExpansionNormalGeo m_rotated_trans_geg_zernike_exp;
    detail::ZernikeCoordinateMultiplier m_multiplier;
    detail::IsotropicAngleIntegratorCore m_integrator_core;
    std::size_t m_dist_order;
};

/**
    @brief Angle integrated regular and transverse Radon transforms with anisotropic response function using the Zernike based Radon transform.
*/
class AnisotropicTransverseAngleIntegrator
{
public:
    AnisotropicTransverseAngleIntegrator() = default;
    AnisotropicTransverseAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());
    
    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    [[nodiscard]] std::size_t
    truncation_order() const noexcept { return m_trunc_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param boosts velocities of the observer frame at different times.
        @param era Earth rotation angles.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, SHExpansionVectorSpan<const std::array<double, 2>> response, std::span<const std::array<double, 3>> boosts, std::span<const double> era, std::span<const double> min_speeds, zest::MDSpan<std::array<double, 2>, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param boost velocity of the observer frame.
        @param era Earth rotation angles.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system. The coordinate systems of `response` and `distribution` are related such that they share the same z-axis, but differ by the angle `era` in the xy-plane. Specifically, `era` is the counterclockwise angle from the x-axis of the `distribution` coordinate system to the x-axis of the `response` coordinate system.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        ZernikeExpansionSpan<const std::array<double, 2>> distribution, SHExpansionVectorSpan<const std::array<double, 2>> response, const std::array<double, 3>& boost, double era, std::span<const double> min_speeds, std::span<std::array<double, 2>> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        SHExpansionVectorSpan<const std::array<double, 2>> response, const std::array<double, 3>& boost, double era, std::span<const double> min_speeds, std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_x;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_y;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_z;
    zest::zt::RealZernikeExpansionNormalGeo m_geg_zernike_exp_r2;
    std::vector<std::array<double, 2>> m_rotated_geg_zernike_exp;
    zest::zt::RealZernikeExpansionNormalGeo m_rotated_trans_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_grids;
    std::vector<double> m_rotated_trans_geg_zernike_grids;
    detail::ZernikeCoordinateMultiplier m_multiplier;
    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    detail::AnisotropicAngleIntegratorCore m_integrator_core;
    std::size_t m_dist_order;
    std::size_t m_resp_order;
    std::size_t m_trunc_order;
};

} // namespace zebra
} // namespace zdm