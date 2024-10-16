#pragma once

#include <cstddef>

template <std::size_t M, typename FieldType>
auto last(FieldType a) noexcept
{
    std::array<typename std::remove_cvref<FieldType>::type::value_type, M> res{};
    for (std::size_t i = 0; i < M; ++i)
        res[i] = a[(a.size() - M) + i];
    return res;
}

template <typename FieldType>
auto prod(FieldType a) noexcept
{
    auto res = a[0];
    for (std::size_t i = 1; i < a.size(); ++i)
        res *= a[i];
    return res;
}

/*
View over contiguous multidimensional data with arbitrary layout in the last dimensions.

Given `SubSpanType` describing contiguous data with arbitrary layout, `MultiSuperSpan` describes a multidimensional array view over such data. For example, if `SubSpanType` describes a triangular data
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
template <typename SubspanType, std::size_t NDIM>
class MultiSuperSpan
{
public:
    using element_type = typename SubspanType::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using ConstView = MultiSuperSpan<typename SubspanType::ConstView, NDIM>;

    constexpr MultiSuperSpan(
        element_type* data, const std::array<std::size_t, NDIM>& extents, std::size_t subspan_size_param) noexcept:
        m_data(data), m_size(prod(extents)*SubspanType::size(subspan_size_param)), m_subspan_size(SubspanType::size(subspan_size_param)), m_subspan_size_param(subspan_size_param), m_extents(extents) {}
    
    constexpr MultiSuperSpan(
        std::span<element_type> span, const std::array<std::size_t, NDIM>& extents, std::size_t subspan_size_param) noexcept:
        m_data(span.data()), m_size(prod(extents)*SubspanType::size(subspan_size_param)), m_subspan_size(SubspanType::size(subspan_size_param)), m_subspan_size_param(subspan_size_param), m_extents(extents) {}

    constexpr MultiSuperSpan(
        data_handle_type data, size_type size, std::size_t subspan_size, std::size_t subspan_size_param, const std::array<std::size_t, NDIM>& extents) noexcept:
        m_data(data), m_size(size), m_subspan_size(subspan_size), m_subspan_size_param(subspan_size_param), m_extents(extents) {}

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(
            m_data, m_size, m_subspan_size, m_subspan_size_param, m_extents);
    }
    
    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_size;
    }
    
    [[nodiscard]] std::size_t subspan_size() const noexcept
    {
        return m_subspan_size;
    }

    [[nodiscard]] std::span<const std::size_t, NDIM> extents() const noexcept
    {
        return m_extents;
    }

    [[nodiscard]] std::span<element_type> flatten() const noexcept
    {
        return std::span<element_type>(m_data, m_size);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == NDIM)
    SubspanType operator()(Ts... inds)
    {
        std::size_t ind = idx(inds...);
        return SubspanType(m_data + ind*m_subspan_size, m_subspan_size_param);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < NDIM)
    MultiSuperSpan<SubspanType, NDIM - sizeof...(Ts)> operator()(Ts... inds)
    {
        std::size_t ind = idx(inds...);
        std::array<std::size_t, NDIM - sizeof...(Ts)> extents = last<NDIM - sizeof...(Ts)>(m_extents);
        return MultiSuperSpan<SubspanType, NDIM - sizeof...(Ts)>(m_data + ind*prod(extents)*m_subspan_size, extents, m_subspan_size_param);
    }

    auto operator[](std::size_t i)
    {
        return (*this)(i);
    }

private:
    template <typename... Ts>
    std::size_t idx(std::size_t ind, Ts... inds)
    {
        return idx_impl<1>(ind, inds...);
    }

    template <std::size_t N, typename... Ts>
    std::size_t idx_impl(std::size_t ind, std::size_t next, Ts... inds)
    {
        if constexpr (N < NDIM)
            return idx_impl<N + 1>(ind*m_extents[N] + next, inds...);
        else
            return ind;
    }

    template <std::size_t N>
    std::size_t idx_impl(std::size_t ind)
    {
        return ind;
    }

    data_handle_type m_data;
    size_type m_size;
    std::size_t m_subspan_size;
    std::size_t m_subspan_size_param;
    std::array<std::size_t, NDIM> m_extents;
};

template <typename SubspanType>
using SuperSpan = MultiSuperSpan<SubspanType, 1>;