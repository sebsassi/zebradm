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

#include <array>
#include <cstddef>
#include <span>

namespace zdm
{

namespace detail
{

template <std::size_t M, typename FieldType>
constexpr auto last(FieldType a) noexcept
{
    std::array<typename std::remove_cvref<FieldType>::type::value_type, M> res{};
    for (std::size_t i = 0; i < M; ++i)
        res[i] = a[(a.size() - M) + i];
    return res;
}

template <typename FieldType>
constexpr auto prod(FieldType a) noexcept
{
    auto res = a[0];
    for (std::size_t i = 1; i < a.size(); ++i)
        res *= a[i];
    return res;
}

} // namespace detail

/**
    @brief Multidimensional view over contiguous data with arbitrary layout in the last dimensions.

    @tparam SubspanType view describing multidimensional data
    @tparam rank number of dimensions in the outer view

    Given `SubspanType` describing contiguous data with arbitrary layout, `MultiSuperSpan`
    describes a multidimensional array view over such data. For example, if `SubspanType` describes
    triangular data.
    ```
    (0,0)
    (1,0) (1,1)
    ```
    then `MultiSuperSpan<SubSpanType, 2>` can be used to describe a 2D array of such data
    ```
    (0,0,0,0)            (0,1,0,0)          
    (0,0,1,0) (0,0,1,1)  (0,1,1,0) (0,1,1,1) 

    (1,0,0,0)            (1,1,0,0)          
    (1,0,1,0) (1,0,1,1)  (1,1,1,0) (1,1,1,1)

    (2,0,0,0)            (2,1,0,0)          
    (2,0,1,0) (2,0,1,1)  (2,1,1,0) (2,1,1,1)
    ```
*/
template <typename SubspanType, std::size_t rank>
class MultiSuperSpan
{
public:
    using element_type = typename SubspanType::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using ConstView = MultiSuperSpan<typename SubspanType::ConstView, rank>;

    constexpr MultiSuperSpan(
        element_type* data, const std::array<std::size_t, rank>& extents,
        std::size_t subspan_size_param) noexcept:
        m_data(data), m_size(prod(extents)*SubspanType::size(subspan_size_param)),
        m_subspan_size(SubspanType::size(subspan_size_param)),
        m_subspan_size_param(subspan_size_param), m_extents(extents) {}

    constexpr MultiSuperSpan(
        std::span<element_type> span, const std::array<std::size_t, rank>& extents,
        std::size_t subspan_size_param) noexcept:
        m_data(span.data()), m_size(prod(extents)*SubspanType::size(subspan_size_param)),
        m_subspan_size(SubspanType::size(subspan_size_param)),
        m_subspan_size_param(subspan_size_param), m_extents(extents) {}

    constexpr MultiSuperSpan(
        data_handle_type data, size_type size, std::size_t subspan_size,
        std::size_t subspan_size_param,
        const std::array<std::size_t, rank>& extents) noexcept:
        m_data(data), m_size(size), m_subspan_size(subspan_size),
        m_subspan_size_param(subspan_size_param), m_extents(extents) {}

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_subspan_size, m_subspan_size_param, m_extents);
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept
    {
        return m_size;
    }

    [[nodiscard]] constexpr std::size_t subspan_size() const noexcept
    {
        return m_subspan_size;
    }

    [[nodiscard]] constexpr std::span<const std::size_t, rank> extents() const noexcept
    {
        return m_extents;
    }

    [[nodiscard]] constexpr std::span<element_type> flatten() const noexcept
    {
        return std::span<element_type>(m_data, m_size);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == rank)
    [[nodiscard]] constexpr SubspanType operator()(Ts... inds) const noexcept
    {
        std::size_t ind = idx(inds...);
        return SubspanType(m_data + ind*m_subspan_size, m_subspan_size_param);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < rank)
    [[nodiscard]] constexpr MultiSuperSpan<SubspanType, rank - sizeof...(Ts)> operator()(Ts... inds) const noexcept
    {
        std::size_t ind = idx(inds...);
        std::array<std::size_t, rank - sizeof...(Ts)> extents = last<rank - sizeof...(Ts)>(m_extents);
        return MultiSuperSpan<SubspanType, rank - sizeof...(Ts)>(m_data + ind*prod(extents)*m_subspan_size, extents, m_subspan_size_param);
    }

    template <typename... Ts>
    [[nodiscard]] constexpr auto operator[](Ts... inds) const noexcept
    {
        return (*this)(inds...);
    }

private:
    template <typename... Ts>
    [[nodiscard]] constexpr std::size_t idx(std::size_t ind, Ts... inds) const noexcept
    {
        return idx_impl<1>(ind, inds...);
    }

    template <std::size_t N, typename... Ts>
    [[nodiscard]] constexpr std::size_t idx_impl(std::size_t ind, std::size_t next, Ts... inds) const noexcept
    {
        if constexpr (N < rank)
            return idx_impl<N + 1>(ind*m_extents[N] + next, inds...);
        else
            return ind;
    }

    template <std::size_t N>
    [[nodiscard]] constexpr std::size_t idx_impl(std::size_t ind) const noexcept
    {
        return ind;
    }

    data_handle_type m_data;
    size_type m_size;
    std::size_t m_subspan_size;
    std::size_t m_subspan_size_param;
    std::array<std::size_t, rank> m_extents;
};

/**
    @brief View over contiguous data with arbitrary layout in the last dimensions.

    @tparam SubspanType view describing multidimensional data

    Given `SubspanType` describing contiguous data with arbitrary layout, `SuperSpan`
    describes a multidimensional array view over such data. For example, if `SubspanType` describes
    triangular data.
    ```
    (0,0)
    (1,0) (1,1)
    ```
    then `SuperSpan<SubSpanType>` can be used to describe a 2D array of such data
    ```
    (0,0,0)          
    (0,1,0) (0,1,1)

    (1,0,0)          
    (1,1,0) (1,1,1)

    (2,0,0)          
    (2,1,0) (2,1,1)
    ```
*/
template <typename SubspanType>
class SuperSpan
{
public:
    using element_type = typename SubspanType::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using ConstView = SuperSpan<typename SubspanType::ConstView>;

    static constexpr size_type size(
        size_type extent, size_type subspan_size_param)
    {
        return extent*SubspanType::size(subspan_size_param);
    }

    constexpr SuperSpan(
        element_type* data, size_type extent, size_type subspan_size_param) noexcept:
        m_data(data), m_size(extent*SubspanType::size(subspan_size_param)),
        m_subspan_size(SubspanType::size(subspan_size_param)),
        m_subspan_size_param(subspan_size_param), m_extent(extent) {}

    constexpr SuperSpan(
        std::span<element_type> span, size_type extent,
        size_type subspan_size_param) noexcept:
        m_data(span.data()), m_size(extent*SubspanType::size(subspan_size_param)),
        m_subspan_size(SubspanType::size(subspan_size_param)),
        m_subspan_size_param(subspan_size_param), m_extent(extent) {}

    constexpr SuperSpan(
        data_handle_type data, size_type size, size_type subspan_size,
        size_type subspan_size_param, size_type extent) noexcept:
        m_data(data), m_size(size), m_subspan_size(subspan_size),
        m_subspan_size_param(subspan_size_param), m_extent(extent) {}

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_subspan_size, m_subspan_size_param, m_extent);
    }

    [[nodiscard]] constexpr size_type size() const noexcept
    {
        return m_size;
    }

    [[nodiscard]] constexpr size_type subspan_size() const noexcept
    {
        return m_subspan_size;
    }

    [[nodiscard]] constexpr size_type extent() const noexcept
    {
        return m_extent;
    }

    [[nodiscard]] constexpr std::span<element_type> flatten() const noexcept
    {
        return std::span<element_type>(m_data, m_size);
    }

    [[nodiscard]] constexpr SubspanType operator()(index_type i) const noexcept
    {
        return SubspanType(m_data + i*m_subspan_size, m_subspan_size_param);
    }

    [[nodiscard]] constexpr auto operator[](index_type i) const noexcept
    {
        return (*this)(i);
    }

private:
    data_handle_type m_data;
    size_type m_size;
    size_type m_subspan_size;
    size_type m_subspan_size_param;
    size_type m_extent;
};

} // namespace zdm
