#pragma once

#include "zest/zernike_glq_transformer.hpp"
#include "zest/sh_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include "linalg.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"
#include "zernike_recursions.hpp"

namespace zebra
{

/**
    @brief Angle integrated Radon transforms using the Zernike based Radon transform.
*/
class IsotropicAngleIntegrator
{
public:
    using DistributionSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>;

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
        DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out);
    
    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        DistributionSpan distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<double> out);
    
private:
    void integrate(
        const Vector<double, 3>& boost, std::span<const double> min_speeds,
        std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_geg_zernike_exp;
    detail::IsotropicAngleIntegratorCore m_integrator_core;
    std::size_t m_dist_order;
};

/**
    @brief Angle integrated Radon transforms with anisotropic response function using the Zernike based Radon transform.
*/
class AnisotropicAngleIntegrator
{
public:
    using DistributionSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>;
    using ResponseSpan = SHExpansionCollectionSpan<const std::array<double, 2>>;

    AnisotropicAngleIntegrator() = default;
    AnisotropicAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);
    
    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boosts velocities of the observer frame at different times.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system. `response` is expressed in a coordinate system whose z-axis is aligned with that of `distribution`, but is rotated around the z-axis by the angle `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, ResponseSpan response, std::span<const double> era, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system. `response` is expressed in a coordinate system whose z-axis is aligned with that of `distribution`, but is rotated around the z-axis by the angle `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        DistributionSpan distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, ResponseSpan response, double era, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        const Vector<double, 3>& boost, std::span<const double> min_speeds, ResponseSpan response, double era, std::size_t geg_order, std::size_t top_order, std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
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
    using DistributionSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>;

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
        DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<std::array<double, 2>, 2> out);
    
    /**
        @brief Angle integrated nontransverse and transverse Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system.
    */
    void integrate(
        DistributionSpan distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<std::array<double, 2>> out);
    
private:
    void integrate(
        const Vector<double, 3>& boost, std::span<const double> min_speeds,
        std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_x;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_y;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_z;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_r2;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_trans_geg_zernike_exp;
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
    using DistributionSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>;
    using ResponseSpan = SHExpansionCollectionSpan<const std::array<double, 2>>;

    AnisotropicTransverseAngleIntegrator() = default;
    AnisotropicTransverseAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);
    
    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boosts velocities of the observer frame at different times.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boost` are expected to be expressed in the same coordinate system. `response` is expressed in a coordinate system whose z-axis is aligned with that of `distribution`, but is rotated around the z-axis by the angle `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, ResponseSpan response, std::span<const double> era, zest::MDSpan<std::array<double, 2>, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity of the observer frame.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of a response function at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note `distribution` and `boosts` are expected to be expressed in the same coordinate system. `response` is expressed in a coordinate system whose z-axis is aligned with that of `distribution`, but is rotated around the z-axis by the angle `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        DistributionSpan distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, ResponseSpan response, double era, std::span<std::array<double, 2>> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        const Vector<double, 3>& boost, std::span<const double> min_speeds, ResponseSpan response, double era, std::size_t geg_order, std::size_t top_order, std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_x;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_y;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_z;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp_r2;
    std::vector<std::array<double, 2>> m_rotated_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_trans_geg_zernike_exp;
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