#pragma once

#include "zest/zernike_glq_transformer.hpp"
#include "zest/sh_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include "linalg.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"

namespace zebra
{

class IsotropicAngleIntegrator
{
public:
    IsotropicAngleIntegrator() = default;
    explicit IsotropicAngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    dist_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boosts array of velocity boost vectors.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_lmax maximum expansion order considered. See notes.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out, std::size_t trunc_lmax = std::numeric_limits<std::size_t>::max());
    
    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity boost vector.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
        @param trunc_lmax maximum expansion order considered. See notes.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<double> out, std::size_t trunc_lmax = std::numeric_limits<std::size_t>::max());
    
private:
    void IsotropicAngleIntegrator::integrate(
        const Vector<double, 3>& boost, std::span<const double> min_speeds,
        std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_geg_zernike_exp;
    detail::IsotropicAngleIntegratorCore m_integrator;
    std::size_t m_dist_order;
};

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
    dist_order() const noexcept { return m_dist_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boosts array of velocity boost vectors.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of response at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_lmax maximum expansion order considered. See notes.

        @note This function assumes that the expansions in `distribution` and `response` have been defined in coordinate systems whose z-axes are aligned, and only differ by the angles expressed in `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, ResponseSpan response, std::span<const double> era, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boost velocity boost vector.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of response at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_lmax maximum expansion order considered. See notes.

        @note This function assumes that the expansions in `distribution` and `response` have been defined in coordinate systems whose z-axes are aligned, and only differ by the angles expressed in `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
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
    detail::AnisotropicAngleIntegratorCore m_integrator;
    std::size_t m_dist_order;
    std::size_t m_resp_order;
    std::size_t m_trunc_order;
};

} // namespace zebra