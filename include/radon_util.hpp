#pragma once

#include <cassert>
#include <span>

#include "zest/zernike_expansion.hpp"
#include "zest/real_sh_expansion.hpp"
#include "zest/sh_glq_transformer.hpp"

#if defined(__GNUC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#endif

namespace util
{

/**
    @brief Check whether two spans overlap
*/
template <typename T, typename S>
    requires std::same_as<std::remove_cv_t<T>, std::remove_cv_t<S>>
[[nodiscard]] constexpr bool
have_overlap(std::span<T> a, std::span<S> b) noexcept
{
    const T* a_begin = a.data();
    const T* a_end = a.data() + a.size();
    const T* b_begin = b.data();
    const T* b_end = b.data() + b.size();
    return std::max(a_begin, b_begin) < std::min(a_end, b_end);
}

// multiply `b` to `a`: `a *= b`
constexpr void
mul(double* RESTRICT a, const double* b, std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] *= b[i];
}

// multiply `b` to `a`: `a *= b`
constexpr void mul(std::span<double> a, std::span<const double> b) noexcept
{
    assert(!have_overlap(a, b));
    const std::size_t size = std::min(a.size(), b.size());
    mul(a.data(), b.data(), size);
}

// multiply `c` and `b` and add to `a`: `a += b*c`
constexpr void fmadd(
    double* RESTRICT a, const double* b, const double* c,
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
    assert(!have_overlap(a, b));
    assert(!have_overlap(a, c));
    assert(b.size() == c.size());
    const std::size_t size = std::min(a.size(), b.size());
    fmadd(a.data(), b.data(), c.data(), size);
}

// multiply `c` and `b` and add to `a`: `a += b*c`
constexpr void fmadd(
    std::array<double, 2>* RESTRICT a, double b, const std::array<double, 2>* c,
    std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
    {
        a[i][0] += b*c[i][0];
        a[i][1] += b*c[i][1];
    }
}

// multiply `c` and `b` and add to `a`: `a += b*c`
constexpr void fmadd(
    std::span<std::array<double, 2>> a, double b, std::span<const std::array<double, 2>> c) noexcept
{
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(a.size(), c.size());
    fmadd(a.data(), b, c.data(), size);
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
constexpr void fmadd(
    double* RESTRICT a, double b, double c, const double* d,
    std::size_t size) noexcept
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] = b + c*d[i];
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
constexpr void fmadd(
    std::span<double> a, double b, double c, std::span<const double> d) noexcept
{
    assert(!have_overlap(a, d));
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

/**
    @brief Apply the transformation `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)` to a Zernike expansion.

    @param in input Zernike expansion `f_{nlm}`
    @param out output Zernike expansion `g_{nlm}`

    @note This transformation appears in the evaluation of the Zernike based Radon transform, where it reduces an expression `f_{nlm}(1 - x^2)C^{3/2}_n(x)` to the form `g_{nlm}P_n(x)` with `g_{nlm}` given as above.
*/
void apply_gegenbauer_reduction(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out) noexcept;

/**
    @brief Apply the transformation `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)` to a Zernike expansion.

    @param exp Zernike expansion

    @note This function expects `f_{nlm}` to be contained in the lower portion of `exp`, such that its order is `exp.order() - 2`, and the rest of the values to be

    @note This transformation appears in the evaluation of the Zernike based Radon transform, where it reduces an expression `f_{nlm}(1 - x^2)C^{3/2}_n(x)` to the form `g_{nlm}P_n(x)` with `g_{nlm}` given as above.
*/
void apply_gegenbauer_reduction_inplace(
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> exp) noexcept;

}