#pragma once

#include <vector>
#include <span>

#include "legendre.hpp"

struct TrapezoidLayout
{
    using index_type = std::size_t;
    [[nodiscard]] static constexpr
    std::size_t idx(std::size_t extra_extent, index_type l, index_type m) noexcept
    {
        return ((l*(l + 1)) >> 1) + l*extra_extent + m;
    }

    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order, std::size_t extra_extent) noexcept
    {
        return ((order*(order + 1)) >> 1) + order*extra_extent;
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t extra_extent, std::size_t l) noexcept
    {
        return extra_extent + l + 1;
    }
};

template <typename ElementType>
class TrapezoidSpan
{
public:
    using Layout = TrapezoidLayout;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;

    static constexpr std::size_t size(
        std::size_t order, std::size_t extra_extent) noexcept
    {
        return Layout::size(order, extra_extent);
    }

    constexpr TrapezoidSpan() = default;
    constexpr TrapezoidSpan(
        element_type* data, std::size_t order, std::size_t extra_extent):
        m_span(data, Layout::size(order, extra_extent)), m_order(order), m_extra_extent(extra_extent) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::size_t
    extra_extent() const noexcept { return m_extra_extent; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return m_span;
    }

    [[nodiscard]] constexpr
    operator TrapezoidSpan<const element_type>() const noexcept
    {
        return TrapezoidSpan<const element_type>(m_span, m_order, m_extra_extent);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type l) const noexcept
    {
        return std::span<element_type>(
                m_span.begin() + Layout::idx(m_extra_extent,l,0), Layout::line_length(m_extra_extent,l));
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](index_type l) const noexcept
    {
        return std::span<element_type>(
                m_span.begin() + Layout::idx(m_extra_extent,l,0), Layout::line_length(m_extra_extent,l));
    }

    [[nodiscard]] constexpr element_type&
    operator()(index_type l, index_type m) const noexcept
    {
        return m_span[Layout::idx(m_extra_extent,l,m)];
    }

private:
    std::span<element_type> m_span;
    std::size_t m_order;
    std::size_t m_extra_extent;
};

class AffineLegendreIntegrals
{
public:
    AffineLegendreIntegrals() = default;
    AffineLegendreIntegrals(
        std::size_t max_order, std::size_t max_extra_extent);
    
    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    void resize(std::size_t max_order, std::size_t max_extra_extent);

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
        double shift, double scale, double half_width, double inv_scale, std::span<const double> affine_legendre, zest::MDSpan<const double, 2> legendre, TrapezoidSpan<double> integrals) noexcept;

    void glq_step(
        double half_width, double inv_scale, std::size_t n, std::span<const double> affine_legendre, zest::MDSpan<const double, 2> legendre, TrapezoidSpan<double> integrals) noexcept;

    void forward_recursion_step(
        double shift, double scale, double half_width, double inv_scale, std::size_t n, std::span<const double> affine_legendre, zest::MDSpan<const double, 2> legendre, TrapezoidSpan<double> integrals) noexcept;
    
    void backward_recursion_step(
        double shift, double scale, double inv_scale, std::size_t n, TrapezoidSpan<double> integrals) noexcept;

    std::vector<double> m_leg_int_top;
    std::vector<double> m_leg_int_bot;
    LegendreIntegralRecursion m_leg_int_rec;
    LegendreArrayRecursion m_affine_legendre;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_nodes;
    std::vector<double> m_legendre;
    std::size_t m_order;
    std::size_t m_extra_extent;
};