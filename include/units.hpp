/*
Copyright (c) 2026 Sebastian Sassi

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
#include <utility>

#include <mp-units/framework.h>
#include <mp-units/systems/hep.h>
#include <mp-units/systems/si.h>
#include <mp-units/math.h>

#include "concepts.hpp"

namespace zdm
{

namespace mpu = mp_units;
using mp_units::quantity;
using mp_units::QuantityOf;
using mp_units::Quantity;
using namespace mp_units::hep;

namespace si = mp_units::si;

QUANTITY_SPEC(momentum_transfer, momentum, mpu::non_negative);
QUANTITY_SPEC(energy_density, energy*number_density);
QUANTITY_SPEC(mass_density, mass*number_density);
QUANTITY_SPEC(energy_differential_rate_per_unit_mass, mpu::inverse(energy*duration*mass));

// Extra time units that are not defined by mp_units::hep
inline constexpr struct minute: mpu::named_unit<"min", mpu::mag<60>*second> {} minute;
inline constexpr struct hour: mpu::named_unit<"h", mpu::mag<60>*minute> {} hour;
inline constexpr struct day: mpu::named_unit<"D", mpu::mag<24>*hour> {} day;
inline constexpr struct Julian_year: mpu::named_unit<"a", mpu::mag_ratio<36'525, 100>*day> {} Julian_year;
inline constexpr struct century: mpu::named_unit<"c", mpu::mag<100>*Julian_year> {} century;
inline constexpr struct millennium: mpu::named_unit<"ka", mpu::mag<1000>*Julian_year> {} millennium;

inline constexpr struct dalton:
    mpu::named_unit<"Da", mpu::mag_ratio<16'605'390'666'050, 10'000'000'000'000>*mpu::mag_power<10, -27>*si::kilo<gram>>
{} dalton;

namespace unit_symbols
{

using namespace mp_units::hep::unit_symbols;

constexpr auto min = minute;
constexpr auto h = hour;
constexpr auto d = day;
constexpr auto a = Julian_year;
constexpr auto ka = millennium;

} // namespace unit_symbols

// Linear algebra operations on vectors and matrices with units.
namespace la
{

template <std::size_t... Inds, Quantity Q>
    requires
        requires (typename Q::rep v) { v.template swizzle<Inds...>(); }
        || requires (typename Q::rep v) { swizzle<Inds...>(v); }
[[nodiscard]] constexpr Quantity auto swizzle(const Q& q) noexcept
{
    if constexpr (requires (typename Q::rep v) { v.template swizzle<Inds...>(); })
        return (q.numerical_value_in(Q::unit).template swizzle<Inds...>())*Q::reference;
    else
        return swizzle<Inds...>(q.numerical_value_in(Q::unit))*Q::reference;
}

template <typename Q1, typename Q2>
    requires static_vector_like<remove_unit<Q1>> && static_vector_like<remove_unit<Q2>>
        && (Quantity<Q1> || Quantity<Q2>)
[[nodiscard]] constexpr mpu::Quantity auto dot(const Q1& q1, const Q2& q2) noexcept
{
    if constexpr (Quantity<Q1> && Quantity<Q2>)
        return dot(q1.numerical_value_in(Q1::unit), q2.numerical_value_in(Q2::unit))*Q1::unit*Q2::unit;
    else if constexpr (Quantity<Q1>)
        return dot(q1, q2.numerical_value_in(Q2::unit))*Q2::unit;
    else
        return dot(q1.numerical_value_in(Q1::unit), q2)*Q1::unit;
}

template <Quantity Q>
    requires static_vector_like<remove_unit<Q>>
[[nodiscard]] constexpr mpu::Quantity auto normalize(const Q& q) noexcept
{
    if constexpr (requires (typename Q::rep v) { v.swizzle(); })
        return (q.numerical_value_in(Q::unit).normalize())*Q::reference;
    else
        return normalize(q.numerical_value_in(Q::unit))*Q::reference;
}

template <typename Q1, Quantity Q2>
    requires static_matrix_like<Q1> && static_vector_like<remove_unit<Q2>>
[[nodiscard]] constexpr Quantity auto matmul(const Q1& q1, const Q2& q2) noexcept
{
    return q1*q2;
}

// NOTE: disabled until mp-units supports rank-2 tensors
//
// template <Quantity Q>
//     requires static_matrix_like<remove_unit<Q>>
// [[nodiscard]] constexpr Quantity auto transpose(const Q& q) noexcept
// {
//     return transpose(q.numerical_value_in(Q::unit))*Q::reference;
// }

template <typename Q1, Quantity Q2>
    requires static_matrix_like<Q1> && static_vector_like<remove_unit<Q2>>
[[nodiscard]] constexpr Quantity auto quadratic_form(const Q1& q1, const Q2& q2)
{
    return dot(q2, matmul(q1, q2));
}

} // namespace la

using mp_units::pow;
using mp_units::sqrt;
using mp_units::cbrt;
using mp_units::exp;
using mp_units::abs;
using mp_units::epsilon;
using mp_units::fma;
using mp_units::fmod;
using mp_units::remainder;
using mp_units::isfinite;
using mp_units::isinf;
using mp_units::isnan;
using mp_units::floor;
using mp_units::ceil;
using mp_units::round;
using mp_units::inverse;
using mp_units::hypot;

} // namespace zdm

namespace zdm::units
{

enum class SIPrefix
{
    Q,
    R,
    Y,
    Z,
    E,
    P,
    T,
    G,
    M, 
    k,
    none,
    m,
    u,
    n,
    p,
    f,
    a,
    z,
    y,
    r,
    q
};

[[nodiscard]] consteval double si_factor(SIPrefix prefix) noexcept
{
    constexpr std::array<double, 21> factors = {
        1.0e+30,
        1.0e+27,
        1.0e+24,
        1.0e+21,
        1.0e+18,
        1.0e+15,
        1.0e+12,
        1.0e+9,
        1.0e+6,
        1.0e+3,
        1.0e+0,
        1.0e-3,
        1.0e-6,
        1.0e-9,
        1.0e-12,
        1.0e-15,
        1.0e-18,
        1.0e-21,
        1.0e-24,
        1.0e-27,
        1.0e-30
    };
    return factors[std::size_t(std::to_underlying(prefix))];
}

[[nodiscard]] consteval double convert(SIPrefix from, SIPrefix to) noexcept
{
    return si_factor(from)/si_factor(to);
}

template <typename From, typename To>
    requires std::same_as<From, To>
[[nodiscard]] consteval double convert_base() noexcept
{
    return 1.0;
}

template <SIPrefix prefix_param, typename BaseUnit>
struct Unit
{
    using base = BaseUnit;
    static constexpr SIPrefix prefix = prefix_param;
};

template <typename Unit1, typename Unit2>
[[nodiscard]] consteval double convert() noexcept
{
    return convert(Unit1::prefix, Unit2::prefix)*convert_base<typename Unit1::base, typename Unit2::base>();
}

template <typename Unit1, typename Unit2>
    requires (!std::same_as<Unit1, Unit2>)
[[nodiscard]] constexpr double convert(double value) noexcept
{
    return convert<Unit1, Unit2>()*value;
}

template <typename Unit1, typename Unit2>
    requires std::same_as<Unit1, Unit2>
[[nodiscard]] constexpr double convert(double value) noexcept
{
    return value;
}

struct eV_base {};

using QeV = Unit<SIPrefix::Q, eV_base>;
using ReV = Unit<SIPrefix::R, eV_base>;
using YeV = Unit<SIPrefix::Y, eV_base>;
using ZeV = Unit<SIPrefix::Z, eV_base>;
using EeV = Unit<SIPrefix::E, eV_base>;
using PeV = Unit<SIPrefix::P, eV_base>;
using TeV = Unit<SIPrefix::T, eV_base>;
using GeV = Unit<SIPrefix::G, eV_base>;
using MeV = Unit<SIPrefix::M, eV_base>;
using keV = Unit<SIPrefix::k, eV_base>;
using eV = Unit<SIPrefix::none, eV_base>;
using meV = Unit<SIPrefix::m, eV_base>;
using ueV = Unit<SIPrefix::u, eV_base>;
using neV = Unit<SIPrefix::n, eV_base>;
using peV = Unit<SIPrefix::p, eV_base>;
using feV = Unit<SIPrefix::f, eV_base>;
using aeV = Unit<SIPrefix::a, eV_base>;
using zeV = Unit<SIPrefix::z, eV_base>;
using yeV = Unit<SIPrefix::y, eV_base>;
using reV = Unit<SIPrefix::r, eV_base>;
using qeV = Unit<SIPrefix::q, eV_base>;

} // namespace zdm::units
