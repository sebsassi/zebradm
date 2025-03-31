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
#include <array>

#include <zest/rotor.hpp>
#include <zest/real_sh_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "linalg.hpp"
#include "affine_legendre.hpp"
#include "affine_legendre_integral.hpp"
#include "zonal_glq_transformer.hpp"
#include "types.hpp"

namespace zdm
{
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
        ZernikeExpansionSpan<const std::array<double, 2>> rotated_geg_zernike_exp, double boost_speed, double min_speed);

    [[nodiscard]] std::array<double, 2> integrate_transverse(
        ZernikeExpansionSpan<const std::array<double, 2>> rotated_geg_zernike_exp,
        ZernikeExpansionSpan<const std::array<double, 2>> rotated_trans_geg_zernike_exp,
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
        zest::st::RealSHSpanGeo<const std::array<double, 2>> response_exp,
        const std::array<double, 3>& boost, double era, double min_speed, 
        zest::WignerdPiHalfCollection wigner_d_pi2);

    [[nodiscard]] std::array<double, 2> integrate_transverse(
        SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_geg_zernike_grids,
        SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_trans_geg_zernike_grids,
        zest::st::RealSHSpanGeo<const std::array<double, 2>> response_exp,
        const std::array<double, 3>& boost, double era, double min_speed, 
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

    ZonalGLQTransformer<zest::st::SHNorm::geo> m_zonal_transformer;
    zest::st::SphereGLQGrid<double> m_rotated_grid;
    std::vector<double> m_rotated_exp;
};

} // namespace detail
} // namespace zebra
} // namespace zdm