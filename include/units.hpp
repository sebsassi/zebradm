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
using mp_units::one;

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

/**
    @brief Apply an arbitrary swizzle operation to a vector quantity.

    Swizzle operations take an arbitrary sequence of vector indices and return
    a new vector given by the components at those indices. More formally, if we
    have a vector \f$\vec{v}\f$ with components \f$v_i\f$ for \f$i \in I\f$,
    where \f$I\f$ is some index set, the swizzled vector \f$\vec{u}\f$ is a
    vector with components \f$v_{\iota(j)}\f$, where \f$i : J \rightarrow I\f$
    is a map from the swizzle index set \f$J\f$ to the vector index set \f$I\f$.

    Swizzles can be used to implement a variety of operations on vectors, such
    as permutations and projections.
*/
template <std::size_t... Inds>
[[nodiscard]] constexpr Quantity auto swizzle(const Quantity auto& q) noexcept
    requires
        requires (typename std::remove_cvref_t<decltype(q)>::rep v) { v.template swizzle<Inds...>(); }
        || requires (typename std::remove_cvref_t<decltype(q)>::rep v) { swizzle<Inds...>(v); }
{
    using rep = std::remove_cvref_t<decltype(q)>::rep;
    if constexpr (requires (rep v) { v.template swizzle<Inds...>(); })
        return (q.numerical_value_in(q.unit).template swizzle<Inds...>())*q.reference;
    else
        return swizzle<Inds...>(q.numerical_value_in(q.unit))*q.reference;
}

/**
    @brief Vector dot product of vector quantities.

    This function evaluates the dot product \f$\vec{q}_1\cdot \vec{q}_2\f$ of
    two vectors \f$\vec{q}_1\f$ and \f$\vec{q}_2\f$, where either one can be a
    quantity.
*/
[[nodiscard]] constexpr Quantity auto
dot(const static_vector_like auto& q1, const static_vector_like auto& q2) noexcept
    requires (Quantity<std::remove_cvref_t<decltype(q1)>> || Quantity<std::remove_cvref_t<decltype(q2)>>)
{
    using Q1 = std::remove_cvref_t<decltype(q1)>;
    using Q2 = std::remove_cvref_t<decltype(q2)>;
    if constexpr (Quantity<Q1> && Quantity<Q2>)
        return dot(q1.numerical_value_in(q1.unit), q2.numerical_value_in(q2.unit))*q1.unit*q2.unit;
    else if constexpr (Quantity<Q1>)
        return dot(q1, q2.numerical_value_in(q2.unit))*q2.unit;
    else
        return dot(q1.numerical_value_in(q1.unit), q2)*q1.unit;
}

/**
    @brief Normalize a vector quantity.

    This function computes the unit vector \f$\hat{q} = \vec{q}/|\vec{q}|\f$
    for a vector quantity \f$\vec{q}\f$. The unit vector is by definition
    a unitless quantity, and therefore in the quantity system carries a unit of
    `one`.
*/
[[nodiscard]] constexpr Quantity auto
normalize(const Quantity auto& q) noexcept
    requires static_vector_like<remove_unit<decltype(q)>>
{
    using rep = std::remove_cvref_t<decltype(q)>::rep;
    if constexpr (requires (rep v) { v.swizzle(); })
        return q.numerical_value_in(q.unit).normalize()*mpu::one;
    else
        return normalize(q.numerical_value_in(q.unit))*mpu::one;
}

/**
    @brief Evaluate the norm of a vector quantity.
*/
[[nodiscard]] constexpr Quantity auto
norm(const Quantity auto& q) noexcept
{
    return mpu::magnitude(q);
}

/**
    @brief Scale the components of a vector quantity by factors from a vector.

    Given a vector quantity \f$\vec{q}\f$ and a vector \f$\vec{s}\f$ of the
    same length, this function returns the vector \f$\vec{q}'\f$ with the
    components \f$\vec{q}'_i = s_iq_i\f$. The vector \f$\vec{s}\f$ may itself
    be a quantity.
*/
[[nodiscard]] constexpr Quantity auto
scale_by(const Quantity auto& q, const static_vector_like auto& s) noexcept
{
    using V = std::remove_cvref_t<decltype(s)>;
    if constexpr (Quantity<V>)
        return scale_by(q.numerical_value_in(q.unit), s.numerical_value_in(s.unit))*q.reference*s.reference;
    else
        return scale_by(q.numerical_value_in(q.unit), s)*q.reference;
}

/**
    @brief Multiply a vector quantity by a matrix.
*/
[[nodiscard]] constexpr Quantity auto
matmul(const static_matrix_like auto& matrix, const Quantity auto& q) noexcept
    requires static_vector_like<remove_unit<decltype(q)>>
{
    return matrix*q;
}

// NOTE: disabled until mp-units supports rank-2 tensors
//
// template <Quantity Q>
//     requires static_matrix_like<remove_unit<Q>>
// [[nodiscard]] constexpr Quantity auto transpose(const Q& q) noexcept
// {
//     return transpose(q.numerical_value_in(Q::unit))*Q::reference;
// }

/**
    @brief Evaluate a quadratic for of a vector quantity.

    Given a matrix \f$M\f$ and a vector quantity \f$\vec{q}\f$, this function
    evaluates the quadratic form \f$Q(\vec{q}) = \vec{q}^\mathsf{T}M\vec{q}\f$.
*/
[[nodiscard]] constexpr Quantity auto
quadratic_form(const static_matrix_like auto& matrix, const Quantity auto& q)
    requires static_vector_like<remove_unit<decltype(q)>>
{
    return dot(q, matmul(matrix, q));
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
