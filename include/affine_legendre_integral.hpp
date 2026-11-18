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
#include "shapes.hpp"

namespace zdm::zebra
{


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
    zdm::legendre::IntegralRecursion m_leg_int_rec;
    zdm::legendre::ArrayRecursion m_affine_legendre;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_nodes;
    std::vector<double> m_legendre;
    std::size_t m_order{};
    std::size_t m_extra_extent{};
};

} // namespace zdm::zebra
