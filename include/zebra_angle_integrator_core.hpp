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
#pragma once

#include <array>
#include <vector>

#include <zest/rotor.hpp>
#include <zest/sh_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "legendre.hpp"
#include "affine_legendre_integral.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "zonal_glq_transformer.hpp"

namespace zdm::zebra::detail
{

/**
    @brief Implements the core angle-integrated weighted offset Zernike-based
    Radon transform.

    @tparam dist_type Angle dependence of distribution.
    @tparam resp_type Angle dependence of response.

    The angle-integrated weighted offset Radon transform () is given by
    \f[
        \overline{\mathcal{R}}_W[f](w,\vec{x}_0)
            \equiv \int_{S^2}W(w,\hat{q})
                \mathcal{R}[f](w + \vec{x}_\text{off}\cdot\hat{q},\hat{q})\,
                d\Omega,
    \f]
    where \f$\mathcal{R}[f](w + \vec{x}_\text{off}\cdot\hat{q},\hat{q})\f$
    is the offset Radon transform of the distribution \f$f(\vec{x})\f$ with
    offset vector \f$\vec{x}_\text{off}\f$, on shell \f$w\f$ with
    weight/response function \f$W(w,\hat{q})\f$.

    This class implements the Zernike-expansion based solution to the above
    formula, given by
    \f[
        \overline{\mathcal{R}}_W[f](w,\vec{x}_\text{off})
            = 2\pi\sum_{n = 0}^N\sum{l = 0}^{n + L'}
                \hat{f}^{(W,R)}_{nl0}(w)A_{nl}(w,x_\text{off}).
    \f]
    Here \f$\hat{f}^{(W,R)}_{nl0}(w)\f$ are the coefficients of the
    Zernike-based Radon transform coefficients of \f$f(\vec{x})\f$ convolved
    with \f$W(w,\hat{q})\f$ under rotation \f$R\f$, and
    \f$A_{nl}(w,x_\text{off})\f$ are the affine Legendre integrals
    \f[
        A_{nl}(w,x_\text{off})
            = \int^{z_\text{max}}_{z_\text{min}}
                P_n(w + x_\text{off}z)P_l(z)\,dz.
    \f]
    The Zernike-based solution simplifies in the cases where either
    \f$f(\vec{x})\f$ or \f$W(w,\hat{q})\f$ is isotropic. Therefore this class
    provides specializations for each combination of isotropic/anisotropic via
    the template parameters.
*/
template <DistType dist_type, RespType resp_type>
class AngleIntegratorCore {};

/**
    @brief Specializaton of the core angle-integrated weighted offset
    Zernike-based Radon transform for isotropic distribution and isotorpic
    weight/response function.
*/
template <>
class AngleIntegratorCore<DistType::iso, RespType::iso>
{
public:
    AngleIntegratorCore() = default;
    explicit AngleIntegratorCore(std::size_t geg_order);

    [[nodiscard]] std::size_t
    order() const noexcept { return m_legendre_integral_recursion.order(); }

    /**
        @brief Resize the integrator.
    */
    void resize(std::size_t geg_order);

    /**
        @brief Evaluate the angle-integrated Radon transform.
    */
    [[nodiscard]] double integrate(
        IsotropicZernikeSpan<const double> geg_zernike_exp, double offset_len, double shell);

    /**
        @brief Evaluate the angle-integrated transverse Radon transform.
    */
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        IsotropicZernikeSpan<const double, 3> trans_geg_zernike_exp, double offset_len, double shell);

private:
    LegendreIntegralRecursion m_legendre_integral_recursion;
    std::vector<la::Vector<double, 2>> m_legendre_integrals;
};

/**
    @brief Specializaton of the core angle-integrated weighted offset
    Zernike-based Radon transform for isotropic distribution and anisotorpic
    weight/response function.
*/
template <>
class AngleIntegratorCore<DistType::iso, RespType::aniso>
{
public:
    AngleIntegratorCore() = default;
    explicit AngleIntegratorCore(std::size_t geg_order, std::size_t resp_order);

    [[nodiscard]] std::size_t
    order() const noexcept { return m_aff_leg_integrals.order(); }

    /**
        @brief Resize the integrator.
    */
    void resize(std::size_t geg_order, std::size_t resp_order);

    /**
        @brief Evaluate the angle-integrated Radon transform.
    */
    [[nodiscard]] double integrate(
        IsotropicZernikeSpan<const double> geg_zernike_exp, SHSpan<const double> response_exp,
        const la::Vector<double, 3>& offset, double rotation_angle, double shell,
        const zest::WignerdPiHalfCollection& wigner_d_pi2);

    /**
        @brief Evaluate the angle-integrated transverse Radon transform.
    */
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        IsotropicZernikeSpan<const double, 3> trans_geg_zernike_exp, SHSpan<const double> response_exp,
        const la::Vector<double, 3>& offset, double rotation_angle, double shell,
        const zest::WignerdPiHalfCollection& wigner_d_pi2);

private:
    void evaluate_aff_leg_ylm_integrals(double shell, double offset_len);

    zest::Rotor m_rotor;
    SHExpansion<double> m_rotated_response_exp;
    std::vector<double> m_zonal_rotated_response_exp;
    AffineLegendreIntegrals m_aff_leg_integrals;
    TrapezoidArray<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integral_norms;
};

/**
    @brief Specializaton of the core angle-integrated weighted offset
    Zernike-based Radon transform for anisotropic distribution and isotorpic
    weight/response function.
*/
template <>
class AngleIntegratorCore<DistType::aniso, RespType::iso>
{
public:
    AngleIntegratorCore() = default;
    explicit AngleIntegratorCore(std::size_t geg_order);

    [[nodiscard]] std::size_t
    order() const noexcept { return m_aff_leg_integrals.order(); }

    /**
        @brief Resize the integrator.
    */
    void resize(std::size_t geg_order);

    /**
        @brief Evaluate the angle-integrated Radon transform.
    */
    [[nodiscard]] double integrate(
        ZernikeSpan<const double> rotated_geg_zernike_exp, double offset_len, double shell);

    /**
        @brief Evaluate the angle-integrated transverse Radon transform.
    */
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        ZernikeSpan<const double> rotated_geg_zernike_exp,
        ZernikeSpan<const double> rotated_trans_geg_zernike_exp,
        double offset_len, double shell);

private:
    void evaluate_aff_leg_ylm_integrals(double shell, double offset_len);

    AffineLegendreIntegrals m_aff_leg_integrals;
    TrapezoidArray<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integral_norms;
};

/**
    @brief Specializaton of the core angle-integrated weighted offset
    Zernike-based Radon transform for anisotropic distribution and anisotorpic
    weight/response function.
*/
template <>
class AngleIntegratorCore<DistType::aniso, RespType::aniso>
{
public:
    AngleIntegratorCore() = default;
    AngleIntegratorCore(
        std::size_t geg_order, std::size_t resp_order, std::size_t top_order);

    [[nodiscard]] zest::Rotor& rotor() { return m_rotor; }

    [[nodiscard]] zest::st::GLQTransformerGeo<>&
    glq_transformer() { return m_glq_transformer; }

    /**
        @brief Resize the integrator.
    */
    void resize(
        std::size_t geg_order, std::size_t resp_order, std::size_t top_order);

    /**
        @brief Evaluate the angle-integrated Radon transform.
    */
    [[nodiscard]] double integrate(
        zest::st::SphereGLQGridVectorSpan<const double> rotated_geg_zernike_grids,
        SHSpan<const double> response_exp,
        const la::Vector<double, 3>& offset, double rotation_angle, double shell,
        const zest::WignerdPiHalfCollection& wigner_d_pi2);

    /**
        @brief Evaluate the angle-integrated transverse Radon transform.
    */
    [[nodiscard]] std::array<double, 2> integrate_transverse(
        zest::st::SphereGLQGridVectorSpan<const double> rotated_geg_zernike_grids,
        zest::st::SphereGLQGridVectorSpan<const double> rotated_trans_geg_zernike_grids,
        SHSpan<const double> response_exp,
        const la::Vector<double, 3>& offset, double rotation_angle, double shell,
        const zest::WignerdPiHalfCollection& wigner_d_pi2);

private:
    void evaluate_aff_leg_ylm_integrals(double shell, double offset_len);

    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    SHExpansion<double> m_rotated_response_exp;
    zest::st::SphereGLQGrid<double> m_rotated_response_grid;

    AffineLegendreIntegrals m_aff_leg_integrals;
    TrapezoidArray<double> m_aff_leg_ylm_integrals;
    std::vector<double> m_ylm_integral_norms;

    ZonalGLQTransformer<zest::st::SHNorm::geo> m_zonal_transformer;
    zest::st::SphereGLQGrid<double> m_rotated_grid;
    std::vector<double> m_rotated_exp;
};

} // namespace zdm::zebra::detail
