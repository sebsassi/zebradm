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

#include <cstddef>
#include <array>

#include <zest/indexing.hpp>
#include <zest/shape.hpp>
#include <zest/triangle_shapes.hpp>
#include <zest/sh_conventions.hpp>

namespace zdm
{

template <std::size_t... inner_extents>
using LegendreShellShape
    = zest::TaggedShape<
        zest::AddInnerTensor<
            zest::AddOuterTensor<
                zest::TriangleShape<zest::Indexing::zero_based, 2>,
                std::dynamic_extent>, inner_extents...>,
        zest::st::Geo>;

template <std::size_t... outer_extents>
using LegendreShellTensorShape
    = zest::TaggedShape<
        zest::AddOuterTensor<
            zest::TriangleShape<zest::Indexing::zero_based, 2>,
            outer_extents..., std::dynamic_extent>,
        zest::st::Geo>;

template <std::size_t... inner_extents>
using IsotropicLegendreShellShape
    = zest::TaggedShape<
        zest::AddInnerTensor<zest::TensorShape<std::dynamic_extent>, inner_extents...>,
        zest::st::Geo>;

template <std::size_t... outer_extents>
using IsotropicLegendreShellTensorShape
    = zest::TaggedShape<
        zest::AddOuterTensor<zest::TensorShape<std::dynamic_extent>, outer_extents...>,
        zest::st::Geo>;

/**
    @brief Shape of a buffer with trapezoidal indexing.

    This class describes the shape of a two-dimensional buffer in the shape of
    a right trapezoid such that the elements are contiguous in memory in the
    (row-major) order
    ```
    (0, 0) (0, 1) (0, 2)
    (1, 0) (1, 2) (1, 3) (1, 4)
    (2, 0) (2, 2) (2, 3) (2, 4) (2, 5)
    ...
    ```
*/
class TrapezoidShape
{
public:
    using size_type = std::size_t;
    using index_type = std::size_t;
    using linear_index_type = std::size_t;
    using index_range = zest::StandardIndexRange<index_type>;
    using extent_type = std::array<size_type, 2>;

    static constexpr size_type rank = 2;
    static constexpr std::size_t linear_extent = std::dynamic_extent;
private:
    template <std::size_t N> struct subshape_helper;

    template <std::size_t N>
        requires (N == 1)
    struct subshape_helper<N> { using type = zest::TensorShape<std::dynamic_extent>; };

    template <std::size_t N>
        requires (N == 2)
    struct subshape_helper<N> { using type = zest::NullShape; };

public:
    /**
        @brief Type of the shape of a lower-dimensional slice.

        @tparam N Number of dimensions to go down by.
    */
    template <std::size_t N>
        requires (0 < N && N < rank)
    using subshape_type = subshape_helper<N>::type;

    constexpr TrapezoidShape() = default;
    explicit constexpr TrapezoidShape(const extent_type& extents):
        m_order{extents[0]}, m_extra_extent{extents[1]}, m_size{size(extents[0], extents[1])} {}

    constexpr TrapezoidShape(size_type order, size_type extra_extent):
        m_order{order}, m_extra_extent{extra_extent}, m_size{size(order, extra_extent)} {}

    /**
        @brief Size of the shape with given parameters.
    */
    [[nodiscard]] static constexpr std::size_t
    size(std::size_t order, std::size_t extra_extent) noexcept
    {
        return ((order*(order + 1)) >> 1) + order*extra_extent;
    }

    /**
        @brief Size of the shape.
    */
    [[nodiscard]] constexpr size_type
    size() const noexcept { return m_size; }

    /**
        @brief Number of rows in the shape.
    */
    [[nodiscard]] constexpr size_type
    order() const noexcept { return m_order; }

    /**
        @brief Number of elements in the first row of the shape.
    */
    [[nodiscard]] constexpr size_type
    extra_extent() const noexcept { return m_extra_extent; }

    /**
        @brief Extent parameters determining the shape.
    */
    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return {m_order, m_extra_extent}; }

    /**
        @brief Get lower dimensional slice.
    */
    [[nodiscard]] constexpr auto
    subshape(index_type l) const noexcept
    {
        return zest::TensorShape<std::dynamic_extent>{m_extra_extent + l + 1};
    }

    /**
        @brief Get lower dimensional slice.
    */
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] index_type l, [[maybe_unused]] index_type m) const noexcept
    {
        return zest::NullShape{};
    }

    /**
        @brief Get linear index.
    */
    [[nodiscard]] constexpr index_type
    operator()(index_type l, index_type m) const noexcept
    {
        assert(m <= l + m_extra_extent);
        return ((l*(l + 1)) >> 1) + l*m_extra_extent + m;
    }

    /**
        @brief Get linear index.
    */
    [[nodiscard]] constexpr index_type
    operator()(index_type l) const noexcept
    {
        return ((l*(l + 1)) >> 1) + l*m_extra_extent;
    }

    /**
        @brief Get index range iterator.
    */
    [[nodiscard]] constexpr index_range
    indices() const noexcept { return {index_type(m_order)}; }

    /**
        @brief Get index range iterator.
    */
    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept
    {
        return {index, index_type(m_order)};
    }

private:
    size_type m_order;
    size_type m_extra_extent;
    size_type m_size;
};

} // namespace zdm
