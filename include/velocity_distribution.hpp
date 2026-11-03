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

#include <zest/zernike_glq_transformer.hpp>

#include "rotation.hpp"
#include "vector.hpp"
#include "types.hpp"
#include "units.hpp"
#include "utility.hpp"

namespace zdm::dist
{

template <typename T>
concept bounded_distribution = requires (const T& dist)
    {
        { dist.normalization() } -> QuantityOf<inverse(pow<3>(speed))>;
        { dist.max_velocity() } -> QuantityOf<speed>;
    };

template <typename T>
concept anisotropic = requires (
        const T& dist, const quantity<velocity[T::velocity_unit], la::Vector<double, 3>>& v)
    {
        { dist(velocity) } -> QuantityOf<inverse(pow<3>(speed))>;
    };

template <typename T>
concept isotropic = requires (
        const T& dist, const quantity<speed[T::velocity_unit], double>& v)
    {
        { dist(velocity) } -> QuantityOf<inverse(pow<3>(speed))>;
    };

template <typename T>
concept bounded_anisotropic_distribution
    = bounded_distribution<T> && anisotropic<T>;

template <typename T>
concept bounded_isotropic_distribution
    = bounded_distribution<T> && isotropic<T>;

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
    operator()(QuantityOf<speed> auto v) noexcept requires isotropic<Distribution>
    {
        return static_cast<double>(
            (util::intpow<3>(m_distribution.max_velocity())*m_distribution(v)).in(one));
    }

    [[nodiscard]] constexpr double
    operator()(QuantityOf<velocity> auto v) noexcept requires anisotropic<Distribution>
    {
        return static_cast<double>(
            (util::intpow<3>(m_distribution.max_velocity())*m_distribution(v)).in(one));
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

template <bounded_anisotropic_distribution Dist>
ZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t lmax)
{
    const typename Dist::speed_type scale = dist.max_velocity();
    auto dist_wrap = [&](const std::array<double, 3>& x)
    {
        return dist(velocity(scale*la::Vector(x)));
    };

    zest::zt::BallGLQGridPoints points(lmax);
    return zest::zt::GLQTransformer<zest::zt::NormedGeo>(lmax)
        .forward_transform(points.generate_values(dist_wrap, lmax), lmax);
}


template <bounded_anisotropic_distribution Dist>
ZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t lmax, const la::RotationMatrix<double, 3>& rotation)
{
    const typename Dist::speed_type scale = dist.max_velocity();
    auto dist_wrap = [&](const std::array<double, 3>& x)
    {
        return dist(rotation*velocity(scale*la::Vector(x)));
    };

    zest::zt::BallGLQGridPoints points(lmax);
    return zest::zt::GLQTransformer<zest::zt::NormedGeo>(lmax)
        .forward_transform(points.generate_values(dist_wrap, lmax), lmax);
}

template <bounded_isotropic_distribution Dist>
IsotropicZernikeExpansion<double> zernike_transform(
    const Scaled<Dist>& dist, std::size_t lmax)
{
    const typename Dist::speed_type scale = dist.max_velocity();
    auto dist_wrap = [&](double x)
    {
        return dist(speed(scale*x));
    };

    zest::zt::RadialGLQGridPoints points(lmax);
    return zest::zt::IsotropicGLQTransformer<zest::zt::NormedGeo>(lmax)
        .forward_transform(points.generate_values(dist_wrap, lmax), lmax);
}

class StandardHaloModel
{
public:
    static constexpr auto velocity_unit = si::kilo<meter>/second;

    using velocity_type = quantity<velocity[velocity_unit], la::Vector<double, 3>>;
    using speed_type = quantity<speed[velocity_unit], double>;

    constexpr StandardHaloModel() = default;
    constexpr StandardHaloModel(
        QuantityOf<speed> auto velocity_dispersion,
        QuantityOf<speed> auto escape_velocity) noexcept:
        m_velocity_dispersion{velocity_dispersion.template in<double>(velocity_unit)},
        m_escape_velocity{escape_velocity.template in<double>(velocity_unit)},
        m_norm{normalization(velocity_dispersion, escape_velocity)} {}

    [[nodiscard]] constexpr quantity<inverse(pow<3>(speed[velocity_unit]))>
    operator()(QuantityOf<speed> auto v) noexcept
    {
        return m_norm*std::exp(-static_cast<double>((v/m_velocity_dispersion).in(zdm::one)));

    }

    [[nodiscard]] constexpr speed_type
    escape_velocity() const noexcept
    {
        return m_escape_velocity;
    }

    [[nodiscard]] constexpr speed_type
    max_velocity() const noexcept
    {
        return m_escape_velocity;
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
    quantity<speed[velocity_unit]> m_escape_velocity;
    quantity<inverse(pow<3>(speed[velocity_unit]))> m_norm;
};

} // namespace zdm
