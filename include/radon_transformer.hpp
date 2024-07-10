#pragma once

#include <stdexcept>
#include <cassert>

#include "zest/zernike_glq_transformer.hpp"
#include "zest/sh_glq_transformer.hpp"
#include "zest/real_ylm.hpp"
#include "zest/rotor.hpp"

#include "affine_legendre.hpp"
#include "linalg.hpp"
#include "multi_span.hpp"
#include "legendre.hpp"

template <typename ElementType>
using SHExpansionSpan = zest::st::RealSHExpansionSpan<ElementType, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

template <typename ElementType>
using SHExpansionCollectionSpan = SuperSpan<SHExpansionSpan<ElementType>>;

class RadonTransformer
{
public:
    RadonTransformer() = default;

    void resize(
        std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order,
        std::size_t min_speeds_size);

    /*
    Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

    Parameters:
    `distribution`: Zernike expansion of the distribution.
    `boosts`: array of velocity boost vectors.
    `min_speeds`: minimum speed parameters of the Radon transform.
    `out`: output values.

    Notes:

    Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void angle_integrated_transform(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out);

    /*
    Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

    Parameters:
    `distribution`: Zernike expansion of the distribution.
    `boosts`: array of velocity boost vectors.
    `min_speeds`: minimum speed parameters of the Radon transform.
    `trunc_lmax`: maximum expansion order considered. See notes.
    `out`: output values.

    Notes:

    Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void angle_integrated_transform(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, std::size_t trunc_lmax, zest::MDSpan<double, 2> out);

    /*
    Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

    Parameters:
    `distribution`: Zernike expansion of the distribution.
    `boosts`: array of velocity boost vectors.
    `min_speeds`: minimum speed parameters of the Radon transform.
    `response`: spherical harmonic expansions of response at `min_speeds`.
    `era`: Earth rotation angles.
    `out`: output values.

    Notes:

    This function assumes that the expansions in `distribution` and `response` have been defined in coordinate systems whose z-axes are aligned, and only differ by the angles expressed in `era`.

    Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void angle_integrated_transform(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, SHExpansionCollectionSpan<const std::array<double, 2>> response, std::span<const double> era, zest::MDSpan<double, 2> out);

    /*
    Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, expressed in terms of its Zernike expansion, combined with an angle-dependent response.

    Parameters:
    `distribution`: Zernike expansion of the distribution.
    `boosts`: array of velocity boost vectors.
    `min_speeds`: minimum speed parameters of the Radon transform.
    `response`: spherical harmonic expansions of response at `min_speeds`.
    `era`: Earth rotation angles.
    `trunc_lmax`: maximum expansion order considered. See notes.
    `out`: output values.

    Notes:

    This function assumes that the expansions in `distribution` and `response` have been defined in coordinate systems whose z-axes are aligned, and only differ by the angles expressed in `era`.

    Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_lmax` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void angle_integrated_transform(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, SHExpansionCollectionSpan<const std::array<double, 2>> response, std::span<const double> era, std::size_t trunc_lmax, zest::MDSpan<double, 2> out);

private:
    [[nodiscard]] static constexpr std::array<std::size_t, 2> geg_top_orders(
        std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
    {
        const std::size_t resp_lmax = resp_order - std::min(1UL, resp_order);
        const std::size_t geg_order = std::min(dist_order + 2, trunc_order);
        const std::size_t geg_lmax = geg_order - std::min(1UL, geg_order);
        const std::size_t top_order
            = std::min(2*geg_lmax + resp_lmax + 1, trunc_order);
        
        return {geg_order, top_order};
    }

    MultiSuperSpan<zest::st::SphereGLQGridSpan<const double>, 2> vmin_contribution_grids(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> geg_zernike_exp, std::span<const double> min_speeds, std::size_t geg_order, std::size_t top_order);

    zest::st::SphereGLQGridSpan<double> 
    combine_vmin_boost_contributions(
        SuperSpan<zest::st::SphereGLQGridSpan<const double>> vmin_cont, 
        const Vector<double, 3>& boost, std::size_t top_order);

    double angle_integral(
        SHExpansionSpan<std::array<double, 2>> radon_coeffs,
        double min_speed, double boost_speed, const Vector<double, 3>& boost);

    zest::TriangleSpan<double, zest::TriangleLayout> evaluate_aff_leg_ylm_integrals(double min_speed, double boost_speed);
    void evaluate_ylm_integrals(double lower_limit, double upper_limit);

    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    zest::zt::ZernikeExpansionOrthoGeo m_rotated_geg_zernike_exp;
    zest::st::RealSHExpansionGeo m_rotated_response_exp;
    std::vector<double> m_rotated_exp;
    std::vector<double> m_geg_zernike_grids;
    std::vector<Vector<double, 3>> m_direction_grid;
    std::vector<double> m_aff_legendre_buffer;
    std::vector<std::array<double, 2>> m_vmin_boost_sep_buffer;
    std::vector<double> m_vmin_part_grid_buffer;
    std::vector<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integrals;
    std::vector<double> m_lower_legendre_integrals;

    zest::st::RealSHExpansionGeo m_radon_coeffs;
    zest::st::RealSHExpansionGeo m_response_expansion;

    zest::st::SphereGLQGrid<double> m_rotated_grid;
    zest::st::SphereGLQGrid<double> m_rotated_response_grid;

    zest::st::SphereGLQGridPoints<> m_grid_points;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    zest::Rotor m_rotor;

    AffineLegendreRecursion m_aff_leg_recursion;
    AffineLegendreIntegralRecursion m_aff_leg_int_rec;
    LegendreArrayRecursion m_leg_recursion;
    LegendreIntegralRecursion m_integral_recursion;
};