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

#include "mp-units/systems/isq.h"
#include "mp-units/systems/si.h"
#include "mp-units/systems/astronomy.h"

#include "concepts.hpp"

namespace zdm
{

namespace mpu = mp_units;
using mp_units::quantity;
using mp_units::QuantityOf;
namespace isq = mp_units::isq;
namespace si = mp_units::si;
namespace ast = mp_units::astronomy;

// Linear algebra operations on vectors and matrices with units.
namespace la
{

template <std::size_t... Inds, mpu::Quantity Q>
    requires
        requires (typename Q::rep v) { v.template swizzle<Inds...>(); }
        || requires (typename Q::rep v) { swizzle<Inds...>(v); }
[[nodiscard]] constexpr mpu::Quantity auto swizzle(const Q& q) noexcept
{
    if constexpr (requires (typename Q::rep v) { v.template swizzle<Inds...>(); })
        return (q.numerical_value_in(Q::unit).template swizzle<Inds...>())*Q::reference;
    else
        return swizzle<Inds...>(q.numerical_value_in(Q::unit))*Q::reference;
}

template <typename Q1, typename Q2>
    requires static_vector_like<remove_unit<Q1>> && static_vector_like<remove_unit<Q2>>
        && (mpu::Quantity<Q1> || mpu::Quantity<Q2>)
[[nodiscard]] constexpr mpu::Quantity auto dot(const Q1& q1, const Q2& q2) noexcept
{
    if constexpr (mpu::Quantity<Q1> && mpu::Quantity<Q2>)
        return dot(q1.numerical_value_in(Q1::unit), q2.numerical_value_in(Q2::unit))*Q1::unit*Q2::unit;
    else if constexpr (mpu::Quantity<Q1>)
        return dot(q1, q2.numerical_value_in(Q2::unit))*Q2::unit;
    else
        return dot(q1.numerical_value_in(Q1::unit), q2)*Q1::unit;
}

template <mpu::Quantity Q>
    requires static_vector_like<remove_unit<Q>>
[[nodiscard]] constexpr mpu::Quantity auto normalize(const Q& q) noexcept
{
    if constexpr (requires (typename Q::rep v) { v.swizzle(); })
        return (q.numerical_value_in(Q::unit).normalize())*Q::reference;
    else
        return normalize(q.numerical_value_in(Q::unit))*Q::reference;
}

template <typename Q1, mpu::Quantity Q2>
    requires static_matrix_like<Q1> && static_vector_like<remove_unit<Q2>>
[[nodiscard]] constexpr mpu::Quantity auto matmul(const Q1& q1, const Q2& q2) noexcept
{
    return q1*q2;
}

// NOTE: disabled until mp-units supports rank-2 tensors
//
// template <mpu::Quantity Q>
//     requires static_matrix_like<remove_unit<Q>>
// [[nodiscard]] constexpr mpu::Quantity auto transpose(const Q& q) noexcept
// {
//     return transpose(q.numerical_value_in(Q::unit))*Q::reference;
// }

template <typename Q1, mpu::Quantity Q2>
    requires static_matrix_like<Q1> && static_vector_like<remove_unit<Q2>>
[[nodiscard]] constexpr mpu::Quantity auto quadratic_form(const Q1& q1, const Q2& q2)
{
    return dot(q2, matmul(q1, q2));
}

} // namespace la

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
