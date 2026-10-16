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
#include "mp-units/systems/si/constants.h"

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

// Additional units
inline constexpr struct dalton:
    mpu::named_unit<"Da", mpu::mag_ratio<16'605'390'666'050, 10'000'000'000'000>*mpu::mag_power<10, -27>*si::kilo<gram>>
{} dalton;

// Numerical constants
inline constexpr struct reduced_plank_constant:
    mpu::named_constant<mpu::symbol_text{u8"\u210f", "hbar"}, planck_constant/(mpu::mag<2>*mpu::pi)>
{} reduced_plank_constant;

namespace codata2022
{

using namespace mp_units::hep::codata2022;

} // namespace codata2022

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
[[nodiscard]] constexpr Quantity auto dot(const Q1& q1, const Q2& q2) noexcept
{
    if constexpr (Quantity<Q1> && Quantity<Q2>)
        return dot(q1.numerical_value_in(Q1::unit), q2.numerical_value_in(Q2::unit))*Q1::unit*Q2::unit;
    else if constexpr (Quantity<Q1>)
        return dot(q1, q2.numerical_value_in(Q2::unit))*Q2::unit;
    else
        return dot(q1.numerical_value_in(Q1::unit), q2)*Q1::unit;
}

[[nodiscard]] constexpr Quantity auto
normalize(const Quantity auto& q) noexcept
    requires static_vector_like<remove_unit<decltype(q)>>
{
    if constexpr (requires (typename decltype(q)::rep v) { v.swizzle(); })
        return (q.numerical_value_in(q.unit).normalize())*q.reference;
    else
        return normalize(q.numerical_value_in(q.unit))*q.reference;
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
