#include "affine_legendre.hpp"

#include <algorithm>
#include <cassert>

AffineLegendreRecursion::AffineLegendreRecursion(std::size_t max_order):
    m_a(max_order), m_b(max_order), m_c(max_order), m_d(max_order),
    m_max_order(max_order)
{
    m_c[1] = 2.0/5.0;
    m_d[1] = 1.0;
    for (std::size_t n = 2; n < max_order; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }
}

void AffineLegendreRecursion::expand(std::size_t max_order)
{
    if (max_order <= m_max_order) return;

    m_a.resize(max_order);
    m_b.resize(max_order);
    m_c.resize(max_order);
    m_d.resize(max_order);

    for (std::size_t n = m_max_order; n < max_order; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }

    m_max_order = max_order;
}

void AffineLegendreRecursion::evaluate_affine(
    zest::TriangleSpan<double, zest::TriangleLayout> expansion, double shift, double scale)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 0) = shift;
    expansion(1, 1) = scale;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*scale*expansion(n - 1, n - 2);
        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_shifted(
    zest::TriangleSpan<double, zest::TriangleLayout> expansion, double shift)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 0) = shift;
    expansion(1, 1) = 1.0;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*expansion(n - 1, n - 2);
        expansion(n, n) = expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_scaled(
    zest::TriangleSpan<double, zest::TriangleLayout> expansion, double scale)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    std::ranges::fill(expansion.flatten(), 0.0);
    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 1) = scale;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1)
                - m_b[n]*expansion(n - 2, 0);
        
        const std::size_t lmin = 1 + ((n & 1) ^ 1);
        for (std::size_t l = lmin; l <= n - 2; l += 2)
        {
            expansion(n, l) = m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1)
                    - m_b[n]*expansion(n - 2, l);
        }

        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

struct ExtraTriangleLayout
{
    using index_type = int;
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l, index_type m) noexcept
    {
         return (((l + 1)*(l + 1)) >> 2) + m;
    }

    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order) noexcept
    {
        return ((order + 1)*(order + 1)) >> 2; 
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return (l >> 1) + 1;
    }
};

void AffineLegendreIntegralRecursion::integrals(
    zest::TriangleSpan<double, zest::TriangleLayout> integrals, double shift, double scale)
{
    if (integrals.order() == 0) return;
    expand(integrals.order());

    if (integrals.order() == 1)
    {
        const double zmin = std::max(-(1.0 + shift)/scale, -1.0);
        const double zmax = std::min((1.0 - shift)/scale, 1.0);
        integrals(0, 0) = zmax - zmin;
        return;
    }

    std::ranges::fill(std::span(integrals.data(), integrals.size()), 0.0);
    if (shift + scale < 1)
    {
        integrals(0, 0) = 2.0;
        integrals(1, 0) = 2.0*shift;
        integrals(1, 1) = (2.0/3.0)*scale;
        for (std::size_t n = 2; n < integrals.order(); ++n)
        {
            integrals(n, 0)
                = (double(2*n - 1)/double(n))*shift*integrals(n - 1, 0)
                - (double(n - 1)/double(n))*integrals(n - 2, 0)
                + (double(2*n - 1)/double(n))*scale*integrals(n - 1, 1);
            for (std::size_t l = 1; l <= n - 2; ++l)
            {
                integrals(n, l)
                    = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                    - (double(n - 1)/double(n))*integrals(n - 2, l)
                    + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                    + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
            }

            integrals(n, n - 1)
                = (double(2*n - 1)/double(n))*shift*integrals(n - 1, n - 1)
                + (double(n - 1)/double(n))*scale*integrals(n - 1, n - 2);
            integrals(n, n)
                = (double(2*n - 1)/double(2*n + 1))*scale*integrals(n - 1, n - 1);
        }
    }
    else if (scale > shift - 1)
    {

        if (scale > shift + 1)
            integrals(0, 0) = 2.0/scale;
        else
            integrals(0, 0) = 1.0 + (1.0 - shift)/scale;

        const std::size_t extent = integrals.order();
        const std::size_t llimit = 2*extent - 1;

        m_leg_int_rec.legendre_integral(m_leg_int_top, (1.0 - shift)/scale);

        if (scale > shift + 1)
            m_leg_int_rec.legendre_integral(m_leg_int_bot, -(1.0 + shift)/scale);
        else
            std::ranges::fill(m_leg_int_bot, 0.0);

        zest::TriangleSpan<double, ExtraTriangleLayout> extra_triangle;

        extra_triangle(0, 0)
            = m_leg_int_top[llimit - 1] - m_leg_int_bot[llimit - 1];
        extra_triangle(1, 0)
            = m_leg_int_top[llimit - 2] - m_leg_int_bot[llimit - 2];
        extra_triangle(2, 0)
            = m_leg_int_top[llimit - 3] - m_leg_int_bot[llimit - 3];

        for (std::size_t k = 1; k < llimit - 1; ++k)
        {
            extra_triangle(k, 0)
                = m_leg_int_top[llimit - 1 - k] - m_leg_int_bot[llimit - 1 - k];
            
            std::size_t l = llimit - k;
            extra_triangle(k, 1) = shift*extra_triangle(k - 1, 0)
                + (double(l + 1)/double(2*l + 1))*scale*extra_triangle(k - 2, 0)
                + (double(l)/double(2*l + 1))*scale*extra_triangle(k, 0);
            
            for (std::size_t j = 2; j < (k/2)*2; ++j)
            {   
                std::size_t n = j;
                std::size_t l = llimit - 1 - k + j;
                extra_triangle(k, j)
                    = (double(2*n - 1)/double(n))*shift*extra_triangle(k - 1, j)
                    - (double(n - 1)/double(n))*extra_triangle(k - 2, j)
                    + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*extra_triangle(k - 1, j)
                    + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*extra_triangle(k, j - 1);
            }
        }

        integrals(1, 0) = shift*integrals(0, 0) + scale*extra_triangle(0, 1);
        integrals(1, 1) = shift*extra_triangle(0, 1) + (2.0/3.0)*scale*extra_triangle(0, 2) + (1.0/3.0)*scale*integrals(0, 0);
        
        for (std::size_t n = 2; n < extent; ++n)
        {
            for (std::size_t l = 0; l < (n/2)*2; ++l)
            {
                integrals(n, l)
                    = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                    - (double(n - 1)/double(n))*integrals(n - 2, l)
                    + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                    + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
            }

            for (std::size_t l = (n/2)*2; l <= n; ++l)
            {
                std::size_t k = llimit - 1 - ((n - l) & 1);
                std::size_t j = n;
                integrals(n, l)
                    = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                    - (double(n - 1)/double(n))*integrals(n - 2, l)
                    + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*extra_triangle(k, j)
                    + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
            }
        }
    }
}