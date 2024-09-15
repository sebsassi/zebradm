#pragma once

#include <cassert>
#include <span>

#include "zest/zernike_expansion.hpp"
#include "zest/real_sh_expansion.hpp"
#include "zest/sh_glq_transformer.hpp"

#define RESTRICT(P) P __restrict__

namespace util
{

// multiply `b` to `a`: `a *= b`
constexpr void
mul(RESTRICT(double*) a, const double* b, std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] *= b[i];
}

// multiply `b` to `a`: `a *= b`
constexpr void mul(std::span<double> a, std::span<const double> b) noexcept
{
    const std::size_t size = std::min(a.size(), b.size());
    mul(a.data(), b.data(), size);
}

// multiply `c` and `b` and add to `a`: `a += b*c`
constexpr void fmadd(
    RESTRICT(double*) a, const double* b, const double* c,
    std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] += b[i]*c[i];
}

// multiply `c` and `b` and add to `a`: `a += b*c`
constexpr void fmadd(
    std::span<double> a, std::span<const double> b,
    std::span<const double> c) noexcept
{
    assert(b.size() == c.size());
    const std::size_t size = std::min(a.size(), b.size());
    fmadd(a.data(), b.data(), c.data(), size);
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
constexpr void fmadd(
    RESTRICT(double*) a, double b, double c, const double* d,
    std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] = b + c*d[i];
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
constexpr void fmadd(
    std::span<double> a, double b, double c, std::span<const double> d) noexcept
{
    const std::size_t size = std::min(a.size(), d.size());
    fmadd(a.data(), b, c, d.data(), size);
}

template <zest::zt::ZernikeNorm NORM>
inline double geg_rec_coeff(std::size_t n) noexcept
{
    if constexpr (NORM == zest::zt::ZernikeNorm::UNNORMED)
        return 1.0/double(2*n + 3);
    else
        return 1.0/std::sqrt(double(2*n + 3));
}

void apply_gegenbauer_reduction(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out) noexcept;

}