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

#include <vector>

#include "types.hpp"
#include "utility.hpp"
#include "moments.hpp"
#include "zebra_radon.hpp"

namespace zdm::zebra::detail
{

/**
    @brief Storage of precomputed values appearing in coefficients in Zernike
    function recursion relations.
*/
class ZernikeRecursionData
{
public:
    ZernikeRecursionData() = default;
    explicit ZernikeRecursionData(std::size_t order);

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    /** @brief Returns `sqrt(n)`. */
    [[nodiscard]] double sqrt_n(std::size_t n) const noexcept
    { return m_sqrt_n[n]; }

    /** @brief Returns `1/sqrt((2*n - 1)*(2*n + 1))`. */
    [[nodiscard]] double inv_sqrt_2nm1_2np1(std::size_t n) const noexcept
    { return m_inv_sqrt_2np1_2np3[n - 1]; }

    /** @brief Returns `1/sqrt((2*n + 1)*(2*n + 3))`. */
    [[nodiscard]] double inv_sqrt_2np1_2np3(std::size_t n) const noexcept
    { return m_inv_sqrt_2np1_2np3[n]; }

    /** @brief Returns `1/sqrt((2*n + 3)*(2*n + 5))`. */
    [[nodiscard]] double inv_sqrt_2np3_2np5(std::size_t n) const noexcept
    { return m_inv_sqrt_2np1_2np3[n + 1]; }

    /** @brief Returns `1/sqrt((2*n + 5)*(2*n + 7))`. */
    [[nodiscard]] double inv_sqrt_2np5_2np7(std::size_t n) const noexcept
    { return m_inv_sqrt_2np1_2np3[n + 2]; }

    void expand(std::size_t order);

private:
    std::vector<double> m_sqrt_n; // sqrt(n)
    std::vector<double> m_inv_sqrt_2np1_2np3; // 1/sqrt((2*n + 1)*(2*n + 3))
    std::size_t m_order{};
};


/**
    @brief Multiply a Zernike expansion by moment.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() <= out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
template <Moment moment>
void multiply_by(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in,
    ZernikeSpan<double> out) noexcept;

template <>
void multiply_by<Moment::identity>(
    [[maybe_unused]] const ZernikeRecursionData& coeff_data,
    [[maybe_unused]] ZernikeSpan<const double> in,
    [[maybe_unused]] ZernikeSpan<double> out) noexcept;


template <>
void multiply_by<Moment::x>(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in,
    ZernikeSpan<double> out) noexcept;

template <>
void multiply_by<Moment::y>(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in,
    ZernikeSpan<double> out) noexcept;

template <>
void multiply_by<Moment::z>(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in,
    ZernikeSpan<double> out) noexcept;

template <>
void multiply_by<Moment::r2>(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in,
    ZernikeSpan<double> out) noexcept;

void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    IsotropicZernikeSpan<const double> in,
    IsotropicZernikeSpan<double> out) noexcept;

/**
    @brief Compute coefficients of a Zernike expansion multiplied by moment and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 <= out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
template <Moment moment>
void multiply_by_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept
{
    assert(in.order() + offset_of(moment) <= out.order());
    multiply_by<moment>(coeff_data, in, out);
    radon_transform_inplace(out);
}

/**
    @brief Compute coefficients of an isotropic Zernike expansion multiplied by
    `r2` and apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 3 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out) noexcept;

void transverse_radon_components(
    IsotropicZernikeSpan<const double> in_radon,
    IsotropicZernikeSpan<const double> in_r2_radon,
    IsotropicZernikeSpan<double, 3> out);

} // namespace zdm::zebra::detail
