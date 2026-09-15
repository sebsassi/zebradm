/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include "types.hpp"
#include "zernike_recursions.hpp"

namespace zdm::zebra
{

enum class MomentCategory: std::uint8_t
{
    identity,
    transverse,
    full
};

enum class Moment: std::uint8_t
{
    identity,
    x,
    y,
    z,
    r2,
    x2,
    y2,
    z2,
    xy,
    xz,
    yz
};

namespace detail
{

[[nodiscard]] consteval std::size_t count_of(MomentCategory category) noexcept
{
    constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
    return counts[std::to_underlying(category)];
}

[[nodiscard]] consteval std::size_t offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    return offsets[std::to_underlying(moment)];
}

template <typename ElementType, DistType dist_type, MomentCategory category>
class RadonMomentSpan:
    public std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeTensorSpan<
            ElementType, zest::zt::NormedGeo, detail::count_of(category)
        >,
        zest::zt::ZernikeTensorSpan<
            ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(category)
        >
    >
{
private:
    using Base = std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeTensorSpan<
            ElementType, zest::zt::NormedGeo, detail::count_of(category)
        >,
        zest::zt::ZernikeTensorSpan<
            ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(category)
        >
    >;

public:
    using Base::Base;

    template <std::integral... Inds>
        requires (sizeof...(Inds) < Base::shape_type::rank)
    [[nodiscard]] auto
    operator[](Moment moment, Inds... indices) const noexcept
    {
        return Base::operator[](std::to_underlying(moment), indices...);
    }
};

template <typename ElementType, DistType dist_type, MomentCategory category>
class RadonMomentArray:
    public std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeExpansionTensor<
            ElementType, zest::zt::NormedGeo, detail::count_of(category)
        >,
        zest::zt::ZernikeExpansionTensor<
            ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(category)
        >
    >
{
private:
    using Base = std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeExpansionTensor<
            ElementType, zest::zt::NormedGeo, detail::count_of(category)
        >,
        zest::zt::ZernikeExpansionTensor<
            ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(category)
        >
    >;

public:
    using Base::Base;

    template <std::integral... Inds>
        requires (sizeof...(Inds) < Base::shape_type::rank)
    [[nodiscard]] auto
    operator[](Moment moment, Inds... indices) const noexcept
    {
        return Base::operator[](std::to_underlying(moment), indices...);
    }
};

} // namespace detail

template <typename ElementType, MomentCategory category>
using RadonMomentSpan = detail::RadonMomentSpan<ElementType, DistType::iso, category>;

template <typename ElementType, MomentCategory category>
using IsotropicRadonMomentSpan = detail::RadonMomentSpan<ElementType, DistType::iso, category>;

template <typename ElementType, MomentCategory category>
using RadonMomentArray = detail::RadonMomentArray<ElementType, DistType::iso, category>;

template <typename ElementType, MomentCategory category>
using IsotropicRadonMomentArray = detail::RadonMomentArray<ElementType, DistType::iso, category>;

class RadonTransformer
{
public:
    RadonTransformer() = default;
    explicit RadonTransformer(std::size_t order): m_recursion_coeffs{order} {}

    template <MomentCategory category>
    void transform(
        IsotropicZernikeSpan<double> zernike_expansion,
        IsotropicRadonMomentSpan<double, category> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[Moment::identity]);
        if constexpr (category != MomentCategory::identity)
        {
            multiply_by_x_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::x]);
            multiply_by_y_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::y]);
            multiply_by_z_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::z]);
            multiply_by_r2_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::r2]);
        }
    }

    template <MomentCategory category>
    [[nodiscard]] IsotropicRadonMomentArray<double, category>
    transform(IsotropicZernikeSpan<double> zernike_expansion)
    {
        IsotropicRadonMomentArray<double, category> moments{zernike_expansion.order()};

        transform(zernike_expansion, moments);
        return moments;
    }

    template <MomentCategory category>
    void transform(
        ZernikeSpan<double> zernike_expansion, RadonMomentSpan<double, category> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[Moment::identity]);
        if constexpr (category != MomentCategory::identity)
        {
            multiply_by_x_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::x]);
            multiply_by_y_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::y]);
            multiply_by_z_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::z]);
            multiply_by_r2_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::r2]);
        }
    }

    template <MomentCategory category>
    [[nodiscard]] RadonMomentArray<double, category>
    transform(ZernikeSpan<double> zernike_expansion)
    {
        RadonMomentArray<double, category> moments{zernike_expansion.order()};
        transform(zernike_expansion, moments);
        return moments;
    }

private:
    detail::ZernikeRecursionData m_recursion_coeffs;
};

} // namespace zdm::zebra
