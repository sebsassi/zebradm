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

#include <span>
#include <vector>

#include <zest/md_span.hpp>
#include <zest/shape.hpp>
#include <zest/shaped_array.hpp>
#include <zest/shaped_span.hpp>

#include "legendre.hpp"

namespace zdm::zebra
{

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

/**
    @brief A non-owning view of data with trapezoidal shape.

    @tparam ElementType Type of elements.
*/
template <typename ElementType>
using TrapezoidSpan = zest::ShapedSpan<ElementType, TrapezoidShape>;

/**
    @brief A container for data with trapezoidal shape.

    @tparam ElementType Type of elements.
*/
template <typename ElementType>
using TrapezoidArray = zest::ShapedArray<ElementType, TrapezoidShape>;

/**
    @brief Class for evaluating affine legendre integrals.

    This class evaluates the affine Legendre integrals
    \f[
        A_{nl}(w,x_\text{off})
            = \int^{z_\text{max}}_{z_\text{min}}
                P_n(w + x_\text{off}z)P_l(z)\,dz
    \f]
    for a range of indices \f$0 \leq l \leq n \leq N\f$.

*/
class AffineLegendreIntegrals
{
public:
    AffineLegendreIntegrals() = default;
    AffineLegendreIntegrals(
        std::size_t order, std::size_t extra_extent);

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    /**
        @brief Resize the object.
    */
    void resize(std::size_t order, std::size_t extra_extent);

    /**
        @brief Evaluate the affine legendre integrals.
    */
    void integrals(
        TrapezoidSpan<double> integrals, double shift, double scale);

private:
    void integrals_full_interval(
        TrapezoidSpan<double> integrals, double shift, double scale);

    void integrals_partial_interval(
        TrapezoidSpan<double> integrals, double shift, double scale);

    void integrals_full_dual_interval(
        TrapezoidSpan<double> integrals, double shift, double scale);

    void first_step(
        double shift, double scale, double half_width, double inv_scale,
        std::span<const double> affine_legendre, zest::DynamicMDSpan<const double, 2> legendre,
        TrapezoidSpan<double> integrals) noexcept;

    void glq_step(
        double half_width, double inv_scale, std::size_t n,
        std::span<const double> affine_legendre, zest::DynamicMDSpan<const double, 2> legendre,
        TrapezoidSpan<double> integrals) noexcept;

    void forward_recursion_step(
        double shift, double scale, double half_width, double inv_scale, std::size_t n,
        std::span<const double> affine_legendre, zest::DynamicMDSpan<const double, 2> legendre,
        TrapezoidSpan<double> integrals) noexcept;

    void backward_recursion_step(
        double shift, double scale, double inv_scale, std::size_t n,
        TrapezoidSpan<double> integrals) noexcept;

    std::vector<double> m_leg_int_top;
    std::vector<double> m_leg_int_bot;
    LegendreIntegralRecursion m_leg_int_rec;
    LegendreArrayRecursion m_affine_legendre;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_nodes;
    std::vector<double> m_legendre;
    std::size_t m_order{};
    std::size_t m_extra_extent{};
};

} // namespace zdm::zebra
