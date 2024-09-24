#pragma once

#include "zest/zernike_glq_transformer.hpp"
#include "zest/sh_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include "affine_legendre.hpp"
#include "linalg.hpp"
#include "multi_span.hpp"
#include "affine_legendre_integral.hpp"
#include "zonal_glq_transformer.hpp"

namespace zebra
{
namespace detail
{

class IsotropicAngleIntegratorCore
{
public:
    IsotropicAngleIntegratorCore() = default;
    explicit IsotropicAngleIntegratorCore(std::size_t geg_order);

    [[nodiscard]] std::size_t
    order() const noexcept { return m_aff_leg_integrals.order(); }

    void resize(std::size_t geg_order);

    [[nodiscard]] double integrate(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> rotated_geg_zernike_exp,
        double boost_speed, double min_speed);

    [[nodiscard]] std::array<double, 2> integrate_transverse(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> rotated_geg_zernike_exp,
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> rotated_trans_geg_zernike_exp,
        double boost_speed, double min_speed);

private:
    TrapezoidSpan<double> evaluate_aff_leg_ylm_integrals(
        double min_speed, double boost_speed, std::size_t geg_order);
    
    AffineLegendreIntegrals m_aff_leg_integrals;
    std::vector<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integral_norms;
};

class AnisotropicAngleIntegratorCore
{
public:
    AnisotropicAngleIntegratorCore() = default;
    AnisotropicAngleIntegratorCore(
        std::size_t geg_order, std::size_t resp_order, std::size_t top_order);

    [[nodiscard]] zest::Rotor& rotor() { return m_rotor; }

    [[nodiscard]] zest::st::GLQTransformerGeo<>&
    glq_transformer() { return m_glq_transformer; }

    void resize(
        std::size_t geg_order, std::size_t resp_order, std::size_t top_order);

    [[nodiscard]] double integrate(
        SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_geg_zernike_grids,
        zest::st::RealSHExpansionSpanGeo<const std::array<double, 2>> response_exp,
        double era, const Vector<double, 3>& boost, double min_speed, 
        zest::WignerdPiHalfCollection wigner_d_pi2);

    [[nodiscard]] std::array<double, 2> integrate_transverse(
        SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_geg_zernike_grids,
        SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_trans_geg_zernike_grids,
        zest::st::RealSHExpansionSpanGeo<const std::array<double, 2>> response_exp,
        double era, const Vector<double, 3>& boost, double min_speed, 
        zest::WignerdPiHalfCollection wigner_d_pi2);

private:
    TrapezoidSpan<double> evaluate_aff_leg_ylm_integrals(
        double min_speed, double boost_speed, std::size_t geg_order, std::size_t resp_order);
    
    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    zest::st::RealSHExpansionGeo m_rotated_response_exp;
    zest::st::SphereGLQGrid<double> m_rotated_response_grid;

    AffineLegendreIntegrals m_aff_leg_integrals;
    std::vector<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integral_norms;

    ZonalGLQTransformer<zest::st::SHNorm::GEO> m_zonal_transformer;
    zest::st::SphereGLQGrid<double> m_rotated_grid;
    std::vector<double> m_rotated_exp;
};

} // namespace detail
} // namespace zebra