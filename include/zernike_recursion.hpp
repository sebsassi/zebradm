#pragma once

#include <vector>
#include <array>

#include "zest/zernike_expansion.hpp"

namespace zebra
{
namespace detail
{

/**
    @brief Storage of precomputed values appearing in coefficients in Zernike function recursion relations.
*/
class ZernikeRecursionData
{
public:
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
    std::size_t m_order;
};

/**
    @brief Multiply Zernike expansion by `x`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_x(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out);

/**
    @brief Multiply Zernike expansion by `y`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_y(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out);

/**
    @brief Multiply Zernike expansion by `z`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_z(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out);

/**
    @brief Multiply Zernike expansion by `r*r`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out);

} // detail
} // zebra