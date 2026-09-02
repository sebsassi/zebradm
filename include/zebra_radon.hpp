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

#include <zest/zernike_conventions.hpp>
#include <zest/zernike_expansion.hpp>

#include "radon_util.hpp"
#include "types.hpp"

namespace zdm::zebra
{

enum class ComponentCategory
{
    identity,
    transverse,
    full
};

namespace detail
{

[[nodiscard]] consteval std::size_t count_of(ComponentCategory components) noexcept
{
    constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
    return counts[std::size_t(std::to_underlying(components))];
}

} // namespace detail

template <zest::zt::ZernikeNorm zernike_norm>
void generate_zernike_radon_coeff(std::span<double> zernike_radon_coeff) noexcept
{
    for (std::size_t n = 0; n < zernike_radon_coeff.size(); ++n)
        zernike_radon_coeff[n] = util::zernike_radon_coeff<zernike_norm>(n);
}

template <zest::zt::ZernikeNorm zernike_norm>
void generate_zernike_radon_coeff(IsotropicZernikeSpan<double> zernike_radon_coeff) noexcept
{
    for (std::size_t n : zernike_radon_coeff.indices())
        zernike_radon_coeff[n] = util::zernike_radon_coeff<zernike_norm>(n);
}

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param in input Zernike expansion coefficients
    @param out output Radon expansion coefficients

    Given the Zernike expansion coefficients \f$ f_{nlm} \f$ of a function \f$ f(\vec{x}) \f$, this
    computes the coefficients in the corresponding expansion of the offset Radon transform
    @f[
    \mathcal{R}[f](w,\hat{n}) = \sum_{nlm}g_{gnlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).
    @f]
    These coefficients are given by the formula
    @f[
    g_{nlm} = \frac{f_{nlm}}{2n + 3} - \frac{f_{n-2,lm}}{2n - 1}.
    @f]
*/
void radon_transform(ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept;

void radon_transform(
    ZernikeSpan<const double> in, ZernikeSpan<double> out,
    std::span<const double> zernike_radon_coeff) noexcept;

void radon_transform(IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out) noexcept;

void radon_transform(
    IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out,
    IsotropicZernikeSpan<const double> zernike_radon_coeff) noexcept;

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param exp Zernike expansion coefficients

    Given the Zernike expansion coefficients \f$ f_{nlm} \f$ of a function \f$ f(\vec{x}) \f$, this
    computes the coefficients in the corresponding expansion of the offset Radon transform
    @f[
    \mathcal{R}[f](w,\hat{n}) = \sum_{nlm}g_{gnlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).
    @f]
    These coefficients are given by the formula
    @f[
    g_{nlm} = \frac{f_{nlm}}{2n + 3} - \frac{f_{n-2,lm}}{2n - 1}.
    @f]
*/
void radon_transform_inplace(ZernikeSpan<double> exp) noexcept;

void radon_transform_inplace(ZernikeSpan<double> exp, std::span<const double> zernike_radon_coeff) noexcept;

void radon_transform_inplace(IsotropicZernikeSpan<double> exp) noexcept;

void radon_transform_inplace(IsotropicZernikeSpan<double> exp, IsotropicZernikeSpan<const double> zernike_radon_coeff) noexcept;

template <DistType dist_type, ComponentCategory components_param>
class RadonComponents
{
public:
    static constexpr ComponentCategory components = components_param;

private:
    using expansion_collection_type = std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeExpansionTensorNormalGeo<
            double, detail::count_of(components)
        >,
        zest::zt::ZernikeExpansionTensorNormalGeo<
            double, zest::IndexingMode::zero_based, detail::count_of(components)
        >
    >;

public:
    using radon_span_type = typename expansion_collection_type::template subspan<1>;
    using const_radon_span_type = typename radon_span_type::const_view;

    RadonComponents() = default;

    [[nodiscard]] radon_span_type identity() noexcept { expansions[0]; }
    [[nodiscard]] radon_span_type x() noexcept requires(components != ComponentCategory::identity) { expansions[1]; }
    [[nodiscard]] radon_span_type y() noexcept requires(components != ComponentCategory::identity) { expansions[2]; }
    [[nodiscard]] radon_span_type z() noexcept requires(components != ComponentCategory::identity) { expansions[3]; }
    [[nodiscard]] radon_span_type r2() noexcept requires(components != ComponentCategory::identity) { expansions[4]; }
    [[nodiscard]] radon_span_type x2() noexcept requires(components == ComponentCategory::full) { expansions[5]; }
    [[nodiscard]] radon_span_type y2() noexcept requires(components == ComponentCategory::full) { expansions[6]; }
    [[nodiscard]] radon_span_type z2() noexcept requires(components == ComponentCategory::full) { expansions[7]; }
    [[nodiscard]] radon_span_type xy() noexcept requires(components == ComponentCategory::full) { expansions[8]; }
    [[nodiscard]] radon_span_type xz() noexcept requires(components == ComponentCategory::full) { expansions[9]; }
    [[nodiscard]] radon_span_type yz() noexcept requires(components == ComponentCategory::full) { expansions[10]; }

    [[nodiscard]] const_radon_span_type identity() const noexcept { expansions[0]; }
    [[nodiscard]] const_radon_span_type x() const noexcept requires(components != ComponentCategory::identity) { expansions[1]; }
    [[nodiscard]] const_radon_span_type y() const noexcept requires(components != ComponentCategory::identity) { expansions[2]; }
    [[nodiscard]] const_radon_span_type z() const noexcept requires(components != ComponentCategory::identity) { expansions[3]; }
    [[nodiscard]] const_radon_span_type r2() const noexcept requires(components != ComponentCategory::identity) { expansions[4]; }
    [[nodiscard]] const_radon_span_type x2() const noexcept requires(components == ComponentCategory::full) { expansions[5]; }
    [[nodiscard]] const_radon_span_type y2() const noexcept requires(components == ComponentCategory::full) { expansions[6]; }
    [[nodiscard]] const_radon_span_type z2() const noexcept requires(components == ComponentCategory::full) { expansions[7]; }
    [[nodiscard]] const_radon_span_type xy() const noexcept requires(components == ComponentCategory::full) { expansions[8]; }
    [[nodiscard]] const_radon_span_type xz() const noexcept requires(components == ComponentCategory::full) { expansions[9]; }
    [[nodiscard]] const_radon_span_type yz() const noexcept requires(components == ComponentCategory::full) { expansions[10]; }

private:
    expansion_collection_type expansions;
};

} // namespace zdm::zebra
