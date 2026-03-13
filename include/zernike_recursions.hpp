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
    @brief Multiply a Zernike expansion by `x`.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_x(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Multiply a Zernike expansion by `y`.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_y(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Multiply a Zernike expansion by `z`.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_z(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Multiply a Zernike expansion by `r*r`.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Multiply an isotropic Zernike expansion by `r*r`.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out) noexcept;

/**
    @brief Compute coefficients of a Zernike expansion multiplied by `x` and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_x_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Compute coefficients of a Zernike expansion multiplied by `y` and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_y_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Compute coefficients of a Zernike expansion multiplied by `z` and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_z_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

/**
    @brief Compute coefficients of a Zernike expansion multiplied by `r2` and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 3 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

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

class ZernikeCoordinateMultiplier
{
public:
    ZernikeCoordinateMultiplier() = default;
    explicit ZernikeCoordinateMultiplier(std::size_t order);

    void expand(std::size_t order);

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `x`.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_x(ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `y`.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_y(ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `z`.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_z(ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `r*r`.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <=
    coeff.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_r2(ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `x` and
    apply the resulting expansion coefficients.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_x_and_radon_transform_inplace(
        ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `y` and
    apply the resulting expansion coefficients.

    @param coeff_data Precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_y_and_radon_transform_inplace(
        ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `z` and
    apply the resulting expansion coefficients.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 2 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_z_and_radon_transform_inplace(
        ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

    /**
    @brief Compute coefficients of a Zernike expansion multiplied by `r2` and
    apply the resulting expansion coefficients.

    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 3 < out.order() <=
    coeff_data.order()`. If `in.order() == 0` no work is done.
    */
    void multiply_by_r2_and_radon_transform_inplace(
        ZernikeSpan<const double> in, ZernikeSpan<double> out) const noexcept;

private:
    ZernikeRecursionData m_coeff_data;
};

void transverse_radon_components(
    IsotropicZernikeSpan<const double> in_radon,
    IsotropicZernikeSpan<const double> in_r2_radon,
    IsotropicZernikeSpan<double, 3> out);

class IsotropicZernikeTransverseRadonHelper
{
public:
    IsotropicZernikeTransverseRadonHelper() = default;
    explicit IsotropicZernikeTransverseRadonHelper(std::size_t order):
        m_coeffs(order + 4)
    {
        constexpr double sqrt7 = 2.6457513110645905905016158;
        constexpr double sqrt11 = 3.316624790355399849114933;
        m_coeffs[0, 0] = 0.0;
        m_coeffs[0, 1] = 0.0;
        m_coeffs[0, 2] = 1.0/(5.0*std::numbers::sqrt3);
        m_coeffs[0, 3] = 2.0/(15.0*sqrt7);
        m_coeffs[0, 4] = 0.0;
        m_coeffs[0, 5] = std::numbers::sqrt3/5.0;
        m_coeffs[0, 6] = 2.0/(5.0*sqrt7);
        m_coeffs[0, 7] = 1.0/std::numbers::sqrt3;

        m_coeffs[2, 0] = 0.0;
        m_coeffs[2, 1] = -5.0/(7.0*std::numbers::sqrt3);
        m_coeffs[2, 2] = 33.0/(105.0*sqrt7);
        m_coeffs[2, 3] = -1.0/(63.0*sqrt11);
        m_coeffs[2, 4] = -std::numbers::sqrt3/5.0;
        m_coeffs[2, 5] = sqrt7/45.0;
        m_coeffs[2, 6] = 4.0/(9.0*sqrt11);
        m_coeffs[2, 7] = 1.0/sqrt7;

        generate_coeffs(4);
    }

    std::size_t order() { return m_coeffs.order(); }

    void expand(std::size_t order)
    {
        const std::size_t old_nmax = util::even_floor(m_coeffs.order() - 1);
        m_coeffs.reshape(order + 4);
        generate_coeffs(old_nmax + 2);
    }

    void evaluate_transverse_components(
        IsotropicZernikeSpan<const double> in,
        IsotropicZernikeSpan<double, 3> out) const noexcept
    {
        if (in.order() == 0) return;

        assert(out.order() >= in.order() + 4);
        out[0, 0] = m_coeffs[0, 2]*in[0];
        out[0, 1] = m_coeffs[0, 5]*in[0];
        out[0, 2] = m_coeffs[0, 7]*in[0];

        out[2, 0] = m_coeffs[2, 1]*in[0];
        out[2, 1] = m_coeffs[2, 4]*in[0];
        out[2, 2] = -m_coeffs[0, 7]*in[0];

        if (in.order() > 2)
        {
            out[0, 0] += m_coeffs[0, 3]*in[2];
            out[0, 1] += m_coeffs[0, 6]*in[2];

            out[2, 0] += m_coeffs[2, 2]*in[2];
            out[2, 1] += m_coeffs[2, 5]*in[2];
            out[2, 2] += m_coeffs[2, 7]*in[2];
        }

        if (in.order() > 4)
        {
            out[2, 0] += m_coeffs[2, 3]*in[4];
            out[2, 1] += m_coeffs[2, 6]*in[4];
        }

        const std::size_t nmax = util::even_floor(in.order() + 3);
        for (std::size_t n = 4; n < nmax - 4; n += 2)
        {
            out[n, 0] = m_coeffs[n, 0]*in[n - 4] + m_coeffs[n, 1]*in[n - 2] + m_coeffs[n, 2]*in[n] + m_coeffs[n, 3]*in[n + 2];
            out[n, 1] = m_coeffs[n, 4]*in[n - 2] + m_coeffs[n, 5]*in[n] + m_coeffs[n, 6]*in[n + 2];
            out[n, 2] = m_coeffs[n, 7]*in[n] - m_coeffs[n - 2, 7]*in[n - 2];
        }

        out[nmax, 0] = m_coeffs[nmax, 0]*in[nmax - 4];
        out[nmax, 1] = 0.0;
        out[nmax, 2] = 0.0;

        if (nmax == 4) return;

        out[nmax - 2, 0] = m_coeffs[nmax - 2, 0]*in[nmax - 6] + m_coeffs[nmax - 2, 1]*in[nmax - 4];
        out[nmax - 2, 1] = m_coeffs[nmax - 2, 4]*in[nmax - 4];
        out[nmax - 2, 2] = -m_coeffs[nmax - 4, 7]*in[nmax - 4];

        if (nmax == 6) return;

        out[nmax - 4, 0] = m_coeffs[nmax - 4, 0]*in[nmax - 8] + m_coeffs[nmax - 4, 1]*in[nmax - 6] + m_coeffs[nmax - 4, 2]*in[nmax - 4];
        out[nmax - 4, 1] = m_coeffs[nmax - 4, 4]*in[nmax - 6] + m_coeffs[nmax - 4, 5]*in[nmax - 4];
        out[nmax - 4, 2] = m_coeffs[nmax - 4, 7]*in[nmax - 4] - m_coeffs[nmax - 6, 7]*in[nmax - 6];
    }

private:
    void generate_coeffs(std::size_t start_index)
    {
        for (std::size_t n : m_coeffs.indices(start_index))
        {
            const auto dn = double(n);
            m_coeffs[n, 0] = (dn - 1.0)*(dn + 2.0)/(std::sqrt(2.0*dn - 5.0)*(2.0*dn - 3.0)*(2.0*dn - 1.0));
            m_coeffs[n, 1] = ((dn - 4.0)*(dn + 1.0) + 1.0)/(std::sqrt(2.0*dn - 1.0)*(2.0*dn - 3.0)*(2.0*dn + 3.0));
            m_coeffs[n, 1] = ((4.0*dn*dn + 3.0)*(dn + 1.0) + 3.0)/(std::sqrt(2.0*dn - 1.0)*(2.0*dn - 3.0)*(2.0*dn - 1.0)*(2.0*dn + 1.0)*(2.0*dn + 3.0));
            m_coeffs[n, 2] = ((2.0*dn + 3.0)*dn*dn + 3.0*dn - 1.0)/(std::sqrt(2.0*dn + 3.0)*(2.0*dn - 1.0)*(2.0*dn + 1.0)*(2.0*dn + 5.0));
            m_coeffs[n, 3] = (1.0 - dn)/(std::sqrt(2.0*dn + 7.0)*(2.0*dn + 3.0)*(2.0*dn + 5.0));
            m_coeffs[n, 4] = -(dn + 1.0)/(std::sqrt(2.0*dn - 1.0)*(2.0*dn + 1.0));
            m_coeffs[n, 5] = std::sqrt(2.0*dn + 3.0)/((2.0*dn + 1.0)*(2.0*dn + 5.0));
            m_coeffs[n, 6] = (dn + 2.0)/(std::sqrt(2.0*dn + 7.0)*(2.0*dn + 5.0));
            m_coeffs[n, 7] = 1.0/std::sqrt(2.0*dn + 3.0);
        }
    }

    IsotropicZernikeExpansion<double, 8> m_coeffs;
};

} // namespace zdm::zebra::detail
