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

#include <zest/zernike_glq_transformer.hpp>

#include "rotation.hpp"
#include "vector.hpp"
#include "types.hpp"
#include "units.hpp"
#include "utility.hpp"

namespace zdm::dist
{

/**
    @brief Concept for checking whether a velocity distribution type represents
    a bounded velocity distribution.

    A bounded velocity distribution is a velocity distribution \f$f(\vec{v})\f$
    that vanishes for velocities with magnitudes \f$v \geq v_\text{max}\f$ for
    some maximum magnitude \f$v_\text{max}\f$. That is, formally, \f$f(\vec{v})
    = f(\vec{v})\Theta(v - v_\text{max})\f$.

    In practice, a bounded velocity distribution implements a member function
    `normalization()` that gives the normalization of the velocity distribution,
    and a member function `max_velocity()` that gives \f$v_\text{max}\f$.
*/
template <typename T>
concept bounded_distribution = requires (const T& dist)
    {
        { dist.normalization() } -> QuantityOf<inverse(pow<3>(speed))>;
        { dist.max_velocity() } -> QuantityOf<speed>;
    };

/**
    @brief Concept for checking whether a velocity distribution is anisotropic.

    A velocity distribution is regarded as anisotropic if it is a function of
    velocity. That is, it provides a call operator that takes as argument a
    vector quantity of velocity.
*/
template <typename T>
concept anisotropic = requires (
        const T& dist, const quantity<velocity[T::velocity_unit], la::Vector<double, 3>>& v)
    {
        { dist(velocity) } -> QuantityOf<inverse(pow<3>(speed))>;
    };

/**
    @brief Concept for checking whether a velocity distribution is anisotropic.

    A velocity distribution is regarded as anisotropic if it is a function of
    only the magnitude of the velocity, and is independent of its direction.
    That is, it provides a call operator that takes as argument a scalar
    quantity of speed.
*/
template <typename T>
concept isotropic = requires (
        const T& dist, const quantity<speed[T::velocity_unit], double>& v)
    {
        { dist(velocity) } -> QuantityOf<inverse(pow<3>(speed))>;
    };

/**
    @brief Concept for checking that a velocity distribution is both bounded
    and anisotropic.
*/
template <typename T>
concept bounded_anisotropic_distribution
    = bounded_distribution<T> && anisotropic<T>;

/**
    @brief Concept for checking that a velocity distribution is both bounded
    and isotropic.
*/
template <typename T>
concept bounded_isotropic_distribution
    = bounded_distribution<T> && isotropic<T>;

/**
    @brief Bounded velocity distribution scaled to units of its maximum
    velocity.

    This class implements a velocity distribution whose units are scaled such
    that the maximum velocity is unity. In practice, this means that
    \f[
        f_\text{scaled}(\vec{u}) = v_\text{max}^3f(v_\text{max}\vec{u}).
    \f]
    The scaled velocity distritribution is then unitless.
*/
template <bounded_distribution Distribution>
class Scaled
{
public:
    static constexpr auto velocity_unit = Distribution::velocity_unit;

    using velocity_type = quantity<velocity[velocity_unit], la::Vector<double, 3>>;
    using speed_type = quantity<speed[velocity_unit], double>;

    constexpr Scaled() = default;
    constexpr Scaled(const Distribution& distribution): m_distribution{distribution} {}

    [[nodiscard]] constexpr double
    operator()(double v) noexcept requires isotropic<Distribution>
    {
        const speed_type vmax = m_distribution.max_velocity();
        return static_cast<double>(
            (util::intpow<3>(vmax)*m_distribution(v*vmax))).in(one));
    }

    [[nodiscard]] constexpr double
    operator()(const la::Vector<double, 3>& v) noexcept requires anisotropic<Distribution>
    {
        const speed_type vmax = m_distribution.max_velocity();
        return static_cast<double>(
            (util::intpow<3>(vmax)*m_distribution(velocity(v*vmax))).in(one));
    }

    [[nodiscard]] constexpr speed_type
    max_velocity() const noexcept
    {
        return m_distribution.max_velocity();
    }

    [[nodiscard]] constexpr const Distribution&
    unscaled() const noexcept
    {
        return m_distribution;
    }

private:
    Distribution m_distribution;
};

/**
    @brief Evaluate the Zernike expansion for a scaled anisotropic velocity
    distribution function.

    @param dist Scaled velocity distribution.
    @param max_order Maximum order of the expansion.

    @return Zernike expansion of the scaled velocity distribution.

    Given a scaled velocity distribution
    \f[
        f_\text{scaled}(\vec{u}) = v_\text{max}f(v_\text{max}\vec{u}),
    \f]
    this function returns the coefficients of the real Zernike expansion
    \f[
        f_\text{scaled}(\vec{u}) = \sum_{nlm} f_{nlm}Z_{nlm}(\vec{u}).
    \f]
*/
template <bounded_anisotropic_distribution Dist>
ZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t max_order)
{
    auto dist_wrap = [&](const std::array<double, 3>& x)
    {
        return dist(la::Vector(x));
    };

    zest::zt::BallGLQGridPoints points(max_order);
    return zest::zt::GLQTransformer<zest::zt::NormedGeo>(max_order)
        .forward_transform(points.generate_values(dist_wrap, max_order), max_order);
}

/**
    @brief Evaluate the Zernike expansion for a scaled anisotropic velocity
    distribution function.

    @param dist Scaled velocity distribution.
    @param max_order Maximum order of the expansion.
    @param rotation A rotation matrix applied to the velocity.

    @return Zernike expansion of the scaled velocity distribution.

    Given a scaled velocity distribution
    \f[
        f_\text{scaled}(\vec{u}) = v_\text{max}f(v_\text{max}R\vec{u}),
    \f]
    where $R$ is a rotation matrix, this function returns the coefficients of
    the real Zernike expansion
    \f[
        f_\text{scaled}(\vec{u}) = \sum_{nlm} f_{nlm}Z_{nlm}(\vec{u}).
    \f]

*/
template <bounded_anisotropic_distribution Dist>
ZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t lmax, const la::RotationMatrix<double, 3>& rotation)
{
    auto dist_wrap = [&](const std::array<double, 3>& x)
    {
        return dist(rotation*la::Vector(x));
    };

    zest::zt::BallGLQGridPoints points(lmax);
    return zest::zt::GLQTransformer<zest::zt::NormedGeo>(lmax)
        .forward_transform(points.generate_values(dist_wrap, lmax), lmax);
}

/**
    @brief Evaluate the Zernike expansion for a scaled isotropic velocity
    distribution function.

    @param dist Scaled velocity distribution.
    @param max_order Maximum order of the expansion.
    @param rotation A rotation matrix applied to the velocity.

    @return Zernike expansion of the scaled velocity distribution.

    Given a scaled velocity distribution
    \f[
        f_\text{scaled}(u) = v_\text{max}f(v_\text{max}u),
    \f]
    where $R$ is a rotation matrix, this function returns the coefficients of
    the real Zernike expansion
    \f[
        f_\text{scaled}(\vec{u}) = \sum_n f_nZ_{n00}(\vec{u}).
    \f]

*/
template <bounded_isotropic_distribution Dist>
IsotropicZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t lmax)
{
    zest::zt::RadialGLQGridPoints points(lmax);
    return zest::zt::IsotropicGLQTransformer<zest::zt::NormedGeo>(lmax)
        .forward_transform(points.generate_values(dist, lmax), lmax);
}

/**
    @brief A spherically truncated isotropic Gaussian velocity distribution.

    A spherically truncated isotropic Gaussian velocity distribution has the
    form
    \f[
        f_\text{STIG}(\vec{v}, v_\text{max}, v_\text{disp})
            = \frac{1}{N_t\sqrt{\pi}v_\text{disp}^3}
                \exp\left(-\frac{v^2}{v_\text{disp}^2}\right)
                \Theta(v - v_\text{max}),
    \f]
    where \f$v_\text{disp}\f$ is the velocity dispersion, and
    \f$v_\text{max}\f$ is the cutoff velocity, and the truncation factor
    \f$N_t\f$ for the normalization is given by
    \f[
        N_t
            = \erf\left(-\frac{v_\text{max}^2}{v_\text{disp}^2}\right)
                - \frac{2}{\sqrt{2}}\frac{v_\text{max}}{v_\text{disp}}
                    \exp\left(-\frac{v_\text{max}}{v_\text{disp}}\right).
    \f]
*/
class STIG
{
public:
    static constexpr auto velocity_unit = si::kilo<meter>/second;

    using velocity_type = quantity<velocity[velocity_unit], la::Vector<double, 3>>;
    using speed_type = quantity<speed[velocity_unit], double>;

    constexpr STIG() = default;
    constexpr STIG(
        QuantityOf<speed> auto velocity_dispersion,
        QuantityOf<speed> auto max_speed) noexcept:
        m_velocity_dispersion{velocity_dispersion.template in<double>(velocity_unit)},
        m_max_speed{max_speed.template in<double>(velocity_unit)},
        m_norm{normalization(velocity_dispersion, max_speed)} {}

    [[nodiscard]] constexpr quantity<inverse(pow<3>(speed[velocity_unit]))>
    operator()(QuantityOf<speed> auto v) noexcept
    {
        return m_norm*std::exp(-static_cast<double>((v/m_velocity_dispersion).in(zdm::one)));

    }

    [[nodiscard]] constexpr speed_type
    escape_velocity() const noexcept
    {
        return m_max_speed;
    }

    [[nodiscard]] constexpr speed_type
    max_velocity() const noexcept
    {
        return m_max_speed;
    }

    [[nodiscard]] constexpr speed_type
    velocity_dispersion() const noexcept
    {
        return m_velocity_dispersion;
    }

    [[nodiscard]] constexpr quantity<inverse(pow<3>(speed[velocity_unit]))>
    normalization() const noexcept
    {
        return m_norm;
    }

private:
    [[nodiscard]] static constexpr quantity<inverse(pow<3>(speed[velocity_unit]))>
    normalization(
        QuantityOf<speed> auto vdisp, QuantityOf<speed> auto vesc) noexcept
    {
        const quantity inv_vdisp = 1.0/vdisp.in(velocity_unit);
        const auto ratio
            = static_cast<double>(
                (vesc.template in<double>()*inv_vdisp.template in<double>()).in(zdm::one));
        constexpr double standard_gaussian_norm
            = util::intpow<3>(std::numbers::inv_sqrtpi/std::numbers::sqrt2);
        const quantity gaussian_norm = standard_gaussian_norm*util::intpow<3>(inv_vdisp);
        const double trunc_term
            = std::erf(ratio) - (2.0*std::numbers::inv_sqrtpi)*ratio*std::exp(-util::intpow<2>(ratio));
        return gaussian_norm/trunc_term;
    }

    quantity<speed[velocity_unit]> m_velocity_dispersion;
    quantity<speed[velocity_unit]> m_max_speed;
    quantity<inverse(pow<3>(speed[velocity_unit]))> m_norm;
};

/**
    @brief Velocity distribution of the Standard Halo Model.

    The dark matter velocity distribution in the Standard Halo model is
    described by a spherically truncated isotropic Gaussian
    \f[
        f_\text{SHM}(\vec{v}, v_\text{esc}, v_\text{circ})
            = f_\text{STIG}(\vec{v}, v_\text{esc}, v_text{circ}),
    \f]
    where \f$f_\text{STIG}\f$ is the distribution function for the spherically
    truncated isotropic Gaussian, \f$v_\text{esc}\f$ is the Milky Way escape
    velocity, and \f$v_\text{circ}\f$ is the local circular speed of the Milky
    Way at the distance of the solar system from the galactic center.

    Since the Standard Halo Model velocity distribution function is just a
    spherically truncated isotropic Gaussian, `StandardHaloModel` likewise
    is a simple alias for the corresponding class `STIG`. See its documentation
    for further details.
*/
using StandardHaloModel = STIG;

} // namespace zdm
