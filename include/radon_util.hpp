#pragma once

#include <cassert>
#include <span>

#include "zest/zernike_conventions.hpp"
#include "zest/rotor.hpp"

#if defined(__GNUC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#endif

namespace zebra
{
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
    @brief Set Euler angles for rotation to a coordinate system whose z-axis is in the direction given by the arguments.

    @note The convention for the Euler angles is that used by `zest::Rotor::rotate`.
*/
template <zest::RotationType TYPE>
constexpr std::array<double, 3> euler_angles_to_align_z(
    double azimuth, double colatitude)
{
    if constexpr (TYPE == zest::RotationType::COORDINATE)
        return {azimuth, colatitude, 0.0};
    else
        return {0.0, -colatitude, std::numbers::pi - azimuth};
}

} // namespace util
} // namespace zebra