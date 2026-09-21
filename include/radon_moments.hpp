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
#include "vector.hpp"
#include "zebra_radon.hpp"

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

[[nodiscard]] consteval std::size_t count_of(DistType dist_type, MomentCategory category) noexcept
{
    if (dist_type == DistType::iso)
        return (category == MomentCategory::identity) ? 1 : 3;
    else
    {
        constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
        return counts[std::to_underlying(category)];
    }
}

[[nodiscard]] consteval std::size_t max_offset(DistType dist_type, MomentCategory category) noexcept
{
    return (category == MomentCategory::identity) ? 0 : 2;
}

[[nodiscard]] consteval std::size_t offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    return offsets[std::to_underlying(moment)];
}

template <typename ElementType, MomentCategory category>
class IsotropicRadonMomentSpan:
    public zest::zt::IsotropicZernikeTensorSpan<
        ElementType, zest::zt::NormedGeo, detail::count_of(DistType::iso, category)
    >
{
private:
    using Base = zest::zt::IsotropicZernikeTensorSpan<
        ElementType, zest::zt::NormedGeo, detail::count_of(DistType::iso, category)
    >;

public:
    IsotropicRadonMomentSpan() = default;
    IsotropicRadonMomentSpan(Base::pointer data, std::size_t order):
        Base{data, order + max_offset(DistType::iso, category)} {}
};

template <typename ElementType, MomentCategory category>
class IsotropicRadonMomentArray:
    public zest::zt::IsotropicZernikeExpansionTensor<
        ElementType, zest::zt::NormedGeo, detail::count_of(DistType::iso, category)
    >
{
private:
    using Base = zest::zt::IsotropicZernikeExpansionTensor<
        ElementType, zest::zt::NormedGeo, detail::count_of(DistType::iso, category)
    >;

public:
    IsotropicRadonMomentArray() = default;
    IsotropicRadonMomentArray(std::size_t order):
        Base{order + max_offset(DistType::iso, category)} {}
};

template <typename ElementType, DistType dist_type, MomentCategory category>
class RadonMomentSpan:
    public zest::zt::ZernikeTensorSpan<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(DistType::aniso, category)
    >
{
private:
    using Base = zest::zt::ZernikeTensorSpan<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(DistType::aniso, category)
    >;

public:
    RadonMomentSpan() = default;
    RadonMomentSpan(std::size_t order):
        Base{order + max_offset(DistType::aniso, category)} {}
};

template <typename ElementType, DistType dist_type, MomentCategory category>
class RadonMomentArray:
    public zest::zt::ZernikeExpansionTensor<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(DistType::aniso, category)
    >
{
private:
    using Base = zest::zt::ZernikeExpansionTensor<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, detail::count_of(DistType::aniso, category)
    >;

public:
    RadonMomentArray() = default;
    RadonMomentArray(std::size_t order):
        Base{order + max_offset(DistType::aniso, category)} {}
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

template <typename ElementType, MomentCategory category>
void evaluate_transverse_radon_transform(
    RadonMomentSpan<ElementType, category> moments, const la::Vector<double, 3>& offset,
    ZernikeSpan<ElementType> transverse_radon_transform)
{
    std::ranges::copy(
        moments[Moment::r2].flatten(),
        transverse_radon_transform.flatten().begin());
    util::fmadd(
        transverse_radon_transform.flatten(),
        la::dot(offset, offset), moments[Moment::identity].flatten());
    util::fmadd(
        transverse_radon_transform.flatten(),
        -2.0*offset[0], moments[Moment::x].flatten());
    util::fmadd(
        transverse_radon_transform.flatten(),
        -2.0*offset[1], moments[Moment::y].flatten());
    util::fmadd(
        transverse_radon_transform.flatten(),
        -2.0*offset[2], moments[Moment::z].flatten());
}

template <DistType dist_type, MomentCategory category>
class RadonTransformer {};

template <>
class RadonTransformer<DistType::iso, MomentCategory::identity>
{
public:
    RadonTransformer() = default;

    static void evaluate_transformed_moments(
        IsotropicZernikeSpan<double> zernike_expansion,
        IsotropicRadonMomentSpan<double, MomentCategory::identity> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[0]);
    }

    [[nodiscard]] static IsotropicRadonMomentArray<double, MomentCategory::identity>
    evaluate_transformed_moments(IsotropicZernikeSpan<double> zernike_expansion)
    {
        IsotropicRadonMomentArray<double, MomentCategory::identity> moments{zernike_expansion.order()};

        transform(zernike_expansion, moments);
        return moments;
    }
};

template <MomentCategory category>
    requires (category != MomentCategory::identity)
class RadonTransformer<DistType::iso, category>
{
public:
    RadonTransformer() = default;
    explicit RadonTransformer(std::size_t order): m_transform_helper{order} {}

    void evaluate_transformed_moments(
        IsotropicZernikeSpan<double> zernike_expansion,
        IsotropicRadonMomentSpan<double, category> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[0]);
        m_transform_helper.evaluate_transverse_components(zernike_expansion, radon_moments);
    }

    [[nodiscard]] IsotropicRadonMomentArray<double, category>
    evaluate_transformed_moments(IsotropicZernikeSpan<double> zernike_expansion)
    {
        IsotropicRadonMomentArray<double, category> moments{zernike_expansion.order()};

        transform(zernike_expansion, moments);
        return moments;
    }

private:
    detail::IsotropicZernikeTransverseRadonHelper m_transform_helper;
};

template <>
class RadonTransformer<DistType::aniso, MomentCategory::identity>
{
public:
    RadonTransformer() = default;

    template <MomentCategory category>
    void evaluate_transformed_moments(
        ZernikeSpan<double> zernike_expansion, RadonMomentSpan<double, MomentCategory::identity> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[Moment::identity]);
    }

    template <MomentCategory category>
    [[nodiscard]] RadonMomentArray<double, MomentCategory::identity>
    evaluate_transformed_moments(ZernikeSpan<double> zernike_expansion)
    {
        RadonMomentArray<double, MomentCategory::identity> moments{zernike_expansion.order()};
        transform(zernike_expansion, moments);
        return moments;
    }
};

template <MomentCategory category>
class RadonTransformer<DistType::aniso, category>
{
public:
    RadonTransformer() = default;
    explicit RadonTransformer(std::size_t order): m_recursion_coeffs{order} {}

    void evaluate_transformed_moments(
        ZernikeSpan<double> zernike_expansion, RadonMomentSpan<double, category> radon_moments)
    {
        radon_transform(zernike_expansion, radon_moments[Moment::identity]);
        multiply_by_x_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::x]);
        multiply_by_y_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::y]);
        multiply_by_z_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::z]);
        multiply_by_r2_and_radon_transform_inplace(zernike_expansion, radon_moments[Moment::r2]);
    }

    [[nodiscard]] RadonMomentArray<double, category>
    evaluate_transformed_moments(ZernikeSpan<double> zernike_expansion)
    {
        RadonMomentArray<double, category> moments{zernike_expansion.order()};
        transform(zernike_expansion, moments);
        return moments;
    }

private:
    detail::ZernikeRecursionData m_recursion_coeffs;
};



} // namespace zdm::zebra
