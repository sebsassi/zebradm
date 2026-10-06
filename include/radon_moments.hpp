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
#pragma once

#include <utility>

#include "types.hpp"
#include "zernike_recursions.hpp"
#include "vector.hpp"
#include "zebra_radon.hpp"
#include "moments.hpp"

namespace zdm
{

template <typename ElementType, MomentSet category>
class IsotropicRadonMomentSpan:
    public zest::zt::IsotropicZernikeTensorSpan<
        ElementType, zest::zt::NormedGeo, detail::moment_count(DistType::iso, category)
    >
{
private:
    using Base = zest::zt::IsotropicZernikeTensorSpan<
        ElementType, zest::zt::NormedGeo, detail::moment_count(DistType::iso, category)
    >;

public:
    constexpr IsotropicRadonMomentSpan() = default;
    constexpr IsotropicRadonMomentSpan(Base::pointer data, std::size_t order) noexcept:
        Base{data, order} {}
    constexpr IsotropicRadonMomentSpan(Base::pointer data, const Base::shape_type& shape) noexcept:
        Base{data, shape} {}

    [[nodiscard]] constexpr Base::size_type
    max_order() const noexcept
    {
        return std::get<1>(Base::extents());
    }

    [[nodiscard]] constexpr Base::size_type
    order(IsoMoment moment) const noexcept
    {
        return max_order() - detail::inverse_offset_of(moment);
    }

    [[nodiscard]] constexpr decltype(auto)
    operator[](IsoMoment moment, std::integral auto... inds) noexcept
        requires (sizeof...(inds) + 1 <= Base::shape_type::rank)
    {
        if constexpr (sizeof...(inds) == 0)
        {
            auto exp = Base::operator[](std::to_underlying(moment));
            return decltype(exp){exp.data(), order(moment)};
        }
        else
        {
            return Base::operator[](std::to_underlying(moment), inds...);
        }
    }
};

template <typename ElementType, MomentSet category>
class RadonMomentSpan:
    public zest::zt::ZernikeTensorSpan<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo,
        detail::moment_count(DistType::aniso, category)
    >
{
private:
    using Base = zest::zt::ZernikeTensorSpan<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo,
        detail::moment_count(DistType::aniso, category)
    >;

public:
    RadonMomentSpan() = default;
    RadonMomentSpan(Base::pointer data, std::size_t order) noexcept:
        Base{data, order} {}
    RadonMomentSpan(Base::pointer data, const Base::shape_type& shape) noexcept:
        Base{data, shape} {}

    [[nodiscard]] constexpr Base::size_type
    max_order() const noexcept
    {
        return std::get<0>(std::get<1>(Base::extents()));
    }

    [[nodiscard]] constexpr Base::size_type
    order(Moment moment) const noexcept
    {
        return max_order() - detail::inverse_offset_of(moment);
    }

    [[nodiscard]] constexpr decltype(auto)
    operator[](Moment moment, std::integral auto... inds) noexcept
        requires (sizeof...(inds) + 1 <= Base::shape_type::rank)
    {
        if constexpr (sizeof...(inds) == 0)
        {
            auto exp = Base::operator[](std::to_underlying(moment));
            return decltype(exp){exp.data(), order(moment)};
        }
        else
            return Base::operator[](std::to_underlying(moment), inds...);
    }
};

template <typename ElementType, MomentSet category>
class IsotropicRadonMomentArray:
    public zest::zt::IsotropicZernikeExpansionTensor<
        ElementType, zest::zt::NormedGeo, detail::moment_count(DistType::iso, category)
    >
{
private:
    using Base = zest::zt::IsotropicZernikeExpansionTensor<
        ElementType, zest::zt::NormedGeo, detail::moment_count(DistType::iso, category)
    >;

public:
    IsotropicRadonMomentArray() = default;
    explicit IsotropicRadonMomentArray(std::size_t order): Base{order} {}

    [[nodiscard]] operator
    IsotropicRadonMomentSpan<typename Base::value_type, category>() noexcept
    {
        return {Base::data(), Base::shape()};
    }

    [[nodiscard]] operator
    IsotropicRadonMomentSpan<const typename Base::value_type, category>() const noexcept
    {
        return {Base::data(), Base::shape()};
    }

    [[nodiscard]] Base::size_type
    max_order() const noexcept
    {
        return std::get<1>(Base::extents());
    }

    [[nodiscard]] Base::size_type
    order(IsoMoment moment) const noexcept
    {
        return max_order() - detail::inverse_offset_of(moment);
    }

    [[nodiscard]] decltype(auto)
    operator[](IsoMoment moment, std::integral auto... inds) noexcept
        requires (sizeof...(inds) + 1 <= Base::shape_type::rank)
    {
        if constexpr (sizeof...(inds) == 0)
        {
            auto exp = Base::operator[](std::to_underlying(moment));
            return decltype(exp){exp.data(), order(moment)};
        }
        else
            return Base::operator[](std::to_underlying(moment), inds...);
    }
};

template <typename ElementType, MomentSet category>
IsotropicRadonMomentSpan(IsotropicRadonMomentArray<ElementType, category>)
    -> IsotropicRadonMomentSpan<std::remove_cv_t<ElementType>, category>;

template <typename ElementType, MomentSet category>
class RadonMomentArray:
    public zest::zt::ZernikeExpansionTensor<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo,
        detail::moment_count(DistType::aniso, category)
    >
{
private:
    using Base = zest::zt::ZernikeExpansionTensor<
        ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo,
        detail::moment_count(DistType::aniso, category)
    >;

public:
    RadonMomentArray() = default;
    explicit RadonMomentArray(std::size_t order): Base{order} {}

    [[nodiscard]] operator
    RadonMomentSpan<typename Base::value_type, category>() noexcept
    {
        return {Base::data(), Base::shape()};
    }

    [[nodiscard]] operator
    RadonMomentSpan<const typename Base::value_type, category>() const noexcept
    {
        return {Base::data(), Base::shape()};
    }

    [[nodiscard]] Base::size_type
    max_order() const noexcept
    {
        return std::get<0>(std::get<1>(Base::extents()));
    }

    [[nodiscard]] Base::size_type
    order(Moment moment) const noexcept
    {
        return max_order() - detail::inverse_offset_of(moment);
    }

    [[nodiscard]] decltype(auto)
    operator[](Moment moment, std::integral auto... inds) noexcept
        requires (sizeof...(inds) + 1 <= Base::shape_type::rank)
    {
        if constexpr (sizeof...(inds) == 0)
        {
            auto exp = Base::operator[](std::to_underlying(moment));
            return decltype(exp){exp.data(), order(moment)};
        }
        else
            return Base::operator[](std::to_underlying(moment), inds...);
    }
};

template <typename ElementType, MomentSet category>
RadonMomentSpan(RadonMomentArray<ElementType, category>&)
    -> RadonMomentSpan<ElementType, category>;

template <typename ElementType, MomentSet category>
RadonMomentSpan(const RadonMomentArray<ElementType, category>&)
    -> RadonMomentSpan<const ElementType, category>;

template <typename ElementType, MomentSet category>
void evaluate_transverse_radon_transform(
    RadonMomentSpan<const ElementType, category> moments, const la::Vector<double, 3>& offset,
    ZernikeSpan<ElementType> transverse_radon_transform)
{
    assert(moments.max_order() <= transverse_radon_transform.order());
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

template <DistType dist_type, MomentSet category>
class RadonTransformer {};

template <>
class RadonTransformer<DistType::iso, MomentSet::identity>
{
public:
    RadonTransformer() = default;

    static void evaluate_transformed_moments(
        IsotropicZernikeSpan<double> zernike_expansion,
        IsotropicRadonMomentSpan<double, MomentSet::identity> radon_moments)
    {
        assert(max_radon_order(MomentSet::identity, zernike_expansion.order()) <= radon_moments.max_order());
        zebra::radon_transform(zernike_expansion, radon_moments[IsoMoment::identity]);
    }

    [[nodiscard]] static IsotropicRadonMomentArray<double, MomentSet::identity>
    evaluate_transformed_moments(IsotropicZernikeSpan<double> zernike_expansion)
    {
        IsotropicRadonMomentArray<double, MomentSet::identity>
        moments{max_radon_order(MomentSet::identity, zernike_expansion.order())};

        evaluate_transformed_moments(zernike_expansion, IsotropicRadonMomentSpan(moments));
        return moments;
    }
};

template <MomentSet category>
    requires (category != MomentSet::identity)
class RadonTransformer<DistType::iso, category>
{
public:
    RadonTransformer() = default;
    explicit RadonTransformer(std::size_t order):
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
        m_coeffs[2, 2] = -15.0/(27.0*sqrt7);
        m_coeffs[2, 3] = -4.0/(63.0*sqrt11);
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

    void evaluate_transformed_moments(
        IsotropicZernikeSpan<double> zernike_expansion,
        IsotropicRadonMomentSpan<double, category> radon_moments)
    {
        assert(max_radon_order(category, zernike_expansion.order()) <= radon_moments.max_order());
        evaluate_transverse_moments_impl(zernike_expansion, radon_moments);
    }

    [[nodiscard]] IsotropicRadonMomentArray<double, category>
    evaluate_transformed_moments(IsotropicZernikeSpan<double> zernike_expansion)
    {
        IsotropicRadonMomentArray<double, category>
        moments{max_radon_order(category, zernike_expansion.order())};

        evaluate_transformed_moments(zernike_expansion, moments);
        return moments;
    }

private:
    void evaluate_transverse_moments_impl(
        IsotropicZernikeSpan<const double> in,
        IsotropicRadonMomentSpan<double, category> out) const noexcept
    {
        if (in.order() == 0) return;

        assert(
            in.order() + 2 <= out.order(IsoMoment::identity)
            && in.order() + 2 <= out.order(IsoMoment::linear)
            && in.order() + 4 <= out.order(IsoMoment::quadratic));

        out[IsoMoment::quadratic, 0] = m_coeffs[0, 2]*in[0];
        out[IsoMoment::linear, 0] = m_coeffs[0, 5]*in[0];
        out[IsoMoment::identity, 0] = m_coeffs[0, 7]*in[0];

        out[IsoMoment::quadratic, 2] = m_coeffs[2, 1]*in[0];
        out[IsoMoment::linear, 2] = m_coeffs[2, 4]*in[0];
        out[IsoMoment::identity, 2] = -m_coeffs[0, 7]*in[0];

        if (in.order() > 2)
        {
            out[IsoMoment::quadratic, 0] += m_coeffs[0, 3]*in[2];
            out[IsoMoment::linear, 0] += m_coeffs[0, 6]*in[2];

            out[IsoMoment::quadratic, 2] += m_coeffs[2, 2]*in[2];
            out[IsoMoment::linear, 2] += m_coeffs[2, 5]*in[2];
            out[IsoMoment::identity, 2] += m_coeffs[2, 7]*in[2];
        }

        if (in.order() > 4)
        {
            out[IsoMoment::quadratic, 2] += m_coeffs[2, 3]*in[4];
            out[IsoMoment::linear, 2] += m_coeffs[2, 6]*in[4];
        }

        const std::size_t nmax = util::even_floor(in.order() + 3);
        for (std::size_t n = 4; n < nmax - 4; n += 2)
        {
            out[IsoMoment::quadratic, n]
                = m_coeffs[n, 0]*in[n - 4] + m_coeffs[n, 1]*in[n - 2] + m_coeffs[n, 2]*in[n]
                    + m_coeffs[n, 3]*in[n + 2];
            out[IsoMoment::linear, n]
                = m_coeffs[n, 4]*in[n - 2] + m_coeffs[n, 5]*in[n] + m_coeffs[n, 6]*in[n + 2];
            out[IsoMoment::identity, n]
                = m_coeffs[n, 7]*in[n] - m_coeffs[n - 2, 7]*in[n - 2];
        }

        out[IsoMoment::quadratic, nmax] = m_coeffs[nmax, 0]*in[nmax - 4];
        out[IsoMoment::linear, nmax] = 0.0;
        out[IsoMoment::identity, nmax] = 0.0;

        if (nmax == 4) return;

        out[IsoMoment::quadratic, nmax - 2]
            = m_coeffs[nmax - 2, 0]*in[nmax - 6] + m_coeffs[nmax - 2, 1]*in[nmax - 4];
        out[IsoMoment::linear, nmax - 2]
            = m_coeffs[nmax - 2, 4]*in[nmax - 4];
        out[IsoMoment::identity, nmax - 2]
            = -m_coeffs[nmax - 4, 7]*in[nmax - 4];

        if (nmax == 6) return;

        out[IsoMoment::quadratic, nmax - 4]
            = m_coeffs[nmax - 4, 0]*in[nmax - 8] + m_coeffs[nmax - 4, 1]*in[nmax - 6]
                + m_coeffs[nmax - 4, 2]*in[nmax - 4];
        out[IsoMoment::linear, nmax - 4]
            = m_coeffs[nmax - 4, 4]*in[nmax - 6] + m_coeffs[nmax - 4, 5]*in[nmax - 4];
        out[IsoMoment::identity, nmax - 4]
            = m_coeffs[nmax - 4, 7]*in[nmax - 4] - m_coeffs[nmax - 6, 7]*in[nmax - 6];
    }

    void generate_coeffs(std::size_t start_index)
    {
        for (std::size_t n : m_coeffs.indices(start_index))
        {
            const auto dn = double(n);
            m_coeffs[n, 0]
                = (dn - 1.0)*(dn + 2.0)/(std::sqrt(2.0*dn - 5.0)*(2.0*dn - 3.0)*(2.0*dn - 1.0));
            m_coeffs[n, 1]
                = (dn*(dn - 3.0) - 3.0)/(std::sqrt(2.0*dn - 1.0)*(2.0*dn - 3.0)*(2.0*dn + 3.0));
            m_coeffs[n, 2]
                = -(dn*(dn + 5.0) + 1.0)/(std::sqrt(2.0*dn + 3.0)*(2.0*dn - 1.0)*(2.0*dn + 5.0));
            m_coeffs[n, 3]
                = -(dn - 1.0)*(dn + 2.0)/(std::sqrt(2.0*dn + 7.0)*(2.0*dn + 3.0)*(2.0*dn + 5.0));
            m_coeffs[n, 4]
                = -(dn + 1.0)/(std::sqrt(2.0*dn - 1.0)*(2.0*dn + 1.0));
            m_coeffs[n, 5]
                = std::sqrt(2.0*dn + 3.0)/((2.0*dn + 1.0)*(2.0*dn + 5.0));
            m_coeffs[n, 6]
                = (dn + 2.0)/(std::sqrt(2.0*dn + 7.0)*(2.0*dn + 5.0));
            m_coeffs[n, 7]
                = 1.0/std::sqrt(2.0*dn + 3.0);
        }
    }

    IsotropicZernikeExpansion<double, 8> m_coeffs;
};

template <>
class RadonTransformer<DistType::aniso, MomentSet::identity>
{
public:
    RadonTransformer() = default;

    static void evaluate_transformed_moments(
        ZernikeSpan<const double> zernike_expansion,
        RadonMomentSpan<double, MomentSet::identity> radon_moments)
    {
        assert(max_radon_order(MomentSet::identity, zernike_expansion.order()) <= radon_moments.max_order());
        zebra::radon_transform(zernike_expansion, radon_moments[Moment::identity]);
    }

    [[nodiscard]] static RadonMomentArray<double, MomentSet::identity>
    evaluate_transformed_moments(ZernikeSpan<const double> zernike_expansion)
    {
        RadonMomentArray<double, MomentSet::identity>
        moments{max_radon_order(MomentSet::identity, zernike_expansion.order())};

        evaluate_transformed_moments(zernike_expansion, RadonMomentSpan(moments));
        return moments;
    }
};

template <MomentSet category>
    requires (category != MomentSet::identity)
class RadonTransformer<DistType::aniso, category>
{
public:
    RadonTransformer() = default;
    explicit RadonTransformer(std::size_t order): m_recursion_coeffs{order} {}

    void evaluate_transformed_moments(
        ZernikeSpan<double> zernike_expansion,
        RadonMomentSpan<double, category> radon_moments)
    {
        assert(max_radon_order(category, zernike_expansion.order()) <= radon_moments.max_order());
        zebra::detail::multiply_by_and_radon_transform_inplace<Moment::identity>(m_recursion_coeffs, zernike_expansion, radon_moments[Moment::identity]);
        zebra::detail::multiply_by_and_radon_transform_inplace<Moment::x>(m_recursion_coeffs, zernike_expansion, radon_moments[Moment::x]);
        zebra::detail::multiply_by_and_radon_transform_inplace<Moment::y>(m_recursion_coeffs, zernike_expansion, radon_moments[Moment::y]);
        zebra::detail::multiply_by_and_radon_transform_inplace<Moment::z>(m_recursion_coeffs, zernike_expansion, radon_moments[Moment::z]);
        zebra::detail::multiply_by_and_radon_transform_inplace<Moment::r2>(m_recursion_coeffs, zernike_expansion, radon_moments[Moment::r2]);
    }

    [[nodiscard]] RadonMomentArray<double, category>
    evaluate_transformed_moments(ZernikeSpan<double> zernike_expansion)
    {
        RadonMomentArray<double, category>
        moments{max_radon_order(category, zernike_expansion.order())};

        evaluate_transformed_moments(zernike_expansion, moments);
        return moments;
    }

private:
    zebra::detail::ZernikeRecursionData m_recursion_coeffs;
};

namespace detail
{

void transverse_radon_moments(
    IsotropicZernikeSpan<const double> in_radon,
    IsotropicZernikeSpan<const double> in_r2_radon,
    IsotropicRadonMomentSpan<double, MomentSet::transverse> out);

} // namespace detail

} // namespace zdm
