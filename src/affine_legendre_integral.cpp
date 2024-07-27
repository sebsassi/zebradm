#include "affine_legendre_integral.hpp"

#include <cassert>

#include "zest/triangle_layout.hpp"
#include "zest/gauss_legendre.hpp"

struct ExtraTriangleLayout
{
    using index_type = std::size_t;
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


AffineLegendreIntegralRecursion::AffineLegendreIntegralRecursion(
    std::size_t order, std::size_t extra_extent):
    m_extra_triangle(ExtraTriangleLayout::size(2*(order - std::min(1UL, order)))),
    m_leg_int_top(2*order - std::min(1UL, order) + extra_extent),
    m_leg_int_bot(2*order - std::min(1UL, order) + extra_extent),
    m_leg_int_rec(2*order - std::min(1UL, order) + extra_extent),
    m_affine_legendre(order + extra_extent/2),
    m_glq_nodes(order + extra_extent/2), m_glq_weights(order + extra_extent/2), 
    m_nodes(order + extra_extent/2), 
    m_legendre((order + extra_extent)*(order + extra_extent/2)), m_order(order),
    m_extra_extent(extra_extent)
{
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(
            m_glq_nodes, m_glq_weights, m_glq_weights.size() & 1);
}

void AffineLegendreIntegralRecursion::resize(
    std::size_t order, std::size_t extra_extent)
{
    const std::size_t forward_order = order;
    if (order != m_order || extra_extent != m_extra_extent)
    {
        m_leg_int_top.resize(2*forward_order - std::min(1UL, forward_order) + extra_extent);
        m_leg_int_bot.resize(2*forward_order - std::min(1UL, forward_order) + extra_extent);
        m_leg_int_rec.expand(2*forward_order - std::min(1UL, forward_order) + extra_extent);
        m_affine_legendre.resize(order + extra_extent/2);
        m_glq_nodes.resize(order + extra_extent/2);
        m_glq_weights.resize(order + extra_extent/2);
        m_nodes.resize(order + extra_extent/2);
        m_legendre.resize((order + extra_extent)*(order + extra_extent/2));

        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, m_glq_weights.size() & 1);
    }

    if (order != m_order)
    {
        m_extra_triangle.resize(ExtraTriangleLayout::size(2*(forward_order - std::min(1UL, forward_order))));
    }

    m_order = order;
    m_extra_extent = extra_extent;
}

void AffineLegendreIntegralRecursion::integrals(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    if (integrals.order() == 0) return;
    resize(integrals.order(), integrals.extra_extent());

    std::ranges::fill(integrals.flatten(), 0.0);
    if (scale + 1.0 < shift) return;

    if (shift + scale <= 1.0)
        integrals_full_interval(integrals, shift, scale);
    else if (1.0 - shift < scale && scale < 1.0 + shift)
        integrals_partial_interval(integrals, shift, scale);
    else if (scale > shift + 1.0)
        integrals_full_dual_interval(integrals, shift, scale);
}

// Operates in the region where `shift + scale <= 1`
// In this region the integrals are zero for `n < l`
void AffineLegendreIntegralRecursion::integrals_full_interval(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    assert(shift + scale <= 1.0);
    integrals(0, 0) = 2.0;
    if (integrals.order() == 1) return;

    integrals(1, 0) = 2.0*shift;
    integrals(1, 1) = (2.0/3.0)*scale;
    for (std::size_t n = 2; n < integrals.order(); ++n)
    {
        // Boundary condition: `integrals(n, l) == 0` for `l < 0`
        integrals(n, 0)
            = (double(2*n - 1)/double(n))*shift*integrals(n - 1, 0)
            - (double(n - 1)/double(n))*integrals(n - 2, 0)
            + (double(2*n - 1)/double(n))*scale*integrals(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            // Full recursion
            integrals(n, l)
                = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                - (double(n - 1)/double(n))*integrals(n - 2, l)
                + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
        }

        // Boundary condition: `integrals(n, l) == 0` for `n < l`
        integrals(n, n - 1)
            = (double(2*n - 1)/double(n))*shift*integrals(n - 1, n - 1)
            + (double(n - 1)/double(n))*scale*integrals(n - 1, n - 2);
        integrals(n, n)
            = (double(2*n - 1)/double(2*n + 1))*scale*integrals(n - 1, n - 1);
    }
}

// Operates in the region where `shift + 1 < scale`
// In this region the integrals are zero for `l < n`
void AffineLegendreIntegralRecursion::integrals_full_dual_interval(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    assert(shift + 1.0 < scale);
    const double new_scale = 1.0/scale;
    const double new_shift = -shift*new_scale;
    integrals(0, 0) = 2.0*new_scale;
    if (integrals.order() == 1) return;

    integrals(1, 1) = (2.0/3.0)*new_scale*new_scale;

    m_leg_int_rec.legendre_integral(m_leg_int_top, (1.0 - shift)/scale);
    m_leg_int_rec.legendre_integral(m_leg_int_bot, -(1.0 + shift)/scale);
    for (std::size_t l = 0; l < integrals.extra_extent() + 1; ++l)
        integrals(0, l) = m_leg_int_top[l] - m_leg_int_bot[l];

    for (std::size_t n = 1; n < integrals.order(); ++n)
    {
        // Boundary condition: `integrals(n, l) == 0` for `l < n`
        integrals(n, n) = (double(2*n - 1)/double(2*n + 1))*new_scale*integrals(n - 1, n - 1);
    }

    if (integrals.extra_extent() > 0)
    {
        for (std::size_t n = 1; n < integrals.order(); ++n)
        {
            // Boundary condition: `integrals(n, l) == 0` for `l < n`
            integrals(n, n + 1) = (double(2*n + 1)/double(n + 1))*new_shift*integrals(n, n) + (double(n)/double(n + 1))*new_scale*integrals(n - 1, n);
        }
    }

    const std::size_t nmax = integrals.order() - 1;
    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        m_nodes[i] = new_scale*m_glq_nodes[i] + new_shift;

    m_affine_legendre.init([&](std::span<double> x){
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            x[i] = shift + scale*m_nodes[i];
    });
    m_affine_legendre.iterate(nmax);

    const std::size_t last_extent
        = integrals.order() + integrals.extra_extent();
    zest::MDSpan<double, 2> legendre(
            m_legendre.data(), {last_extent, m_glq_weights.size()});
    legendre_recursion_vec(legendre, m_nodes);

    for (std::size_t k = 2; k < integrals.extra_extent() + 1; ++k)
    {
        for (std::size_t n = 1; n < nmax; ++n)
        {
            // Full recursion
            const std::size_t l = n + k;
            integrals(n, l) = (double(2*l - 1)/double(l))*new_shift*integrals(n, l - 1) - (double(l - 1)/double(l))*integrals(n, l - 2) + (double(2*l - 1)/double(l))*(double(n + 1)/double(2*n + 1))*new_scale*integrals(n + 1, l - 1) + (double(2*l - 1)/double(l))*(double(n)/double(2*n + 1))*new_scale*integrals(n - 1, l - 1);
        }

        // Elements at the `n == nmax` boundary are computed via direct integration
        const std::size_t l = nmax + k;
        double res = 0.0;
        for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
            res += m_glq_weights[i]*m_affine_legendre.current()[i]*legendre(l, i);
        integrals(nmax, l) = new_scale*res;
    }
}

void AffineLegendreIntegralRecursion::integrals_partial_interval_forward(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    const std::size_t order = std::min(integrals.order(), max_forward_order);
    if (order == 0) return;

    m_leg_int_rec.legendre_integral(m_leg_int_top, (1.0 - shift)/scale);

    if (scale > shift + 1.0)
        m_leg_int_rec.legendre_integral(m_leg_int_bot, -(1.0 + shift)/scale);
    else
        std::ranges::fill(m_leg_int_bot, 0.0);

    for (std::size_t l = 0; l < integrals.extra_extent() + 1; ++l)
        integrals(0, l) = m_leg_int_top[l] - m_leg_int_bot[l];

    if (order == 1) return;

    const std::size_t llimit
        = 2*order - 1 + integrals.extra_extent();

    zest::TriangleSpan<double, ExtraTriangleLayout> extra_triangle(m_extra_triangle.data(), 2*(order - 1));

    extra_triangle(0, 0)
        = m_leg_int_top[llimit - 1] - m_leg_int_bot[llimit - 1];
    extra_triangle(1, 0)
        = m_leg_int_top[llimit - 2] - m_leg_int_bot[llimit - 2];

    for (std::size_t k = 2; k < extra_triangle.order(); ++k)
    {
        extra_triangle(k, 0)
            = m_leg_int_top[llimit - 1 - k] - m_leg_int_bot[llimit - 1 - k];
        
        std::size_t l = llimit - k;
        extra_triangle(k, 1) = shift*extra_triangle(k - 1, 0)
            + (double(l + 1)/double(2*l + 1))*scale*extra_triangle(k - 2, 0)
            + (double(l)/double(2*l + 1))*scale*extra_triangle(k, 0);
        
        for (std::size_t j = 2; j < 1 + k/2; ++j)
        {   
            std::size_t n = j;
            std::size_t l = llimit - 1 - k + j;
            extra_triangle(k, j)
                = (double(2*n - 1)/double(n))*shift*extra_triangle(k - 1, j - 1)
                - (double(n - 1)/double(n))*extra_triangle(k - 2, j - 2)
                + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*extra_triangle(k - 2, j - 1)
                + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*extra_triangle(k, j - 1);
        }
    }

    if (integrals.extra_extent() == 0)
        integrals(1, 0) = shift*integrals(0, 0)
            + scale*extra_triangle(extra_triangle.order() - 1, 0);
    else
        integrals(1, 0) = shift*integrals(0, 0) + scale*integrals(0, 1);

    const std::size_t extent = integrals.extra_extent() + 2;
    const std::size_t inner_extent = extent - 2;

    for (std::size_t l = 1; l < inner_extent; ++l)
    {
        integrals(1, l)
            = shift*integrals(0, l)
                + (double(l + 1)/double(2*l + 1))*scale*integrals(0, l + 1)
                + (double(l)/double(2*l + 1))*scale*integrals(0, l - 1);
    }

    if (inner_extent > 0)
    {
        integrals(1, inner_extent)
            = shift*integrals(0, inner_extent)
                + (double(inner_extent + 1)/double(2*inner_extent + 1))*scale*extra_triangle(extra_triangle.order() - 1, 0)
                + (double(inner_extent)/double(2*inner_extent + 1))*scale*integrals(0, inner_extent - 1);
    }
        
    integrals(1, inner_extent + 1)
        = shift*extra_triangle(extra_triangle.order() - 1, 0)
            + (double(inner_extent + 2)/double(2*inner_extent + 3))*scale*extra_triangle(extra_triangle.order() - 2, 0)
            + (double(inner_extent + 1)/double(2*inner_extent + 3))*scale*integrals(0, inner_extent);
    
    for (std::size_t n = 2; n < order; ++n)
    {
        const std::size_t extent = n + integrals.extra_extent() + 1;
        const std::size_t inner_extent = extent - 2;

        integrals(n, 0)
            = (double(2*n - 1)/double(n))*shift*integrals(n - 1, 0)
                - (double(n - 1)/double(n))*integrals(n - 2, 0)
                + (double(2*n - 1)/double(n))*scale*integrals(n - 1, 1);

        for (std::size_t l = 1; l < inner_extent; ++l)
        {
            integrals(n, l)
                = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                    - (double(n - 1)/double(n))*integrals(n - 2, l)
                    + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                    + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
        }

        if (inner_extent > 0)
        {
            integrals(n, inner_extent)
                = (double(2*n - 1)/double(n))*shift*integrals(n - 1, inner_extent)
                    - (double(n - 1)/double(n))*extra_triangle(extra_triangle.order() - 1, n - 2)
                    + (double(2*n - 1)/double(n))*(double(inner_extent + 1)/double(2*inner_extent + 1))*scale*extra_triangle(extra_triangle.order() - 1, n - 1)
                    + (double(2*n - 1)/double(n))*(double(inner_extent)/double(2*inner_extent + 1))*scale*integrals(n - 1, inner_extent - 1);
        }
        
        integrals(n, inner_extent + 1)
            = (double(2*n - 1)/double(n))*shift*extra_triangle(extra_triangle.order() - 1, n - 1)
                - (double(n - 1)/double(n))*extra_triangle(extra_triangle.order() - 2, n - 2)
                + (double(2*n - 1)/double(n))*(double(inner_extent + 2)/double(2*inner_extent + 3))*scale*extra_triangle(extra_triangle.order() - 2, n - 1)
                + (double(2*n - 1)/double(n))*(double(inner_extent + 1)/double(2*inner_extent + 3))*scale*integrals(n - 1, inner_extent);
    }
}

void AffineLegendreIntegralRecursion::integrals_partial_interval_backward(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    if constexpr (max_forward_order == 0)
        if (integrals.order() == 0) return;

    const double zmin = std::max(-1.0, -(1.0 + shift)/scale);
    const double zmax = (1.0 - shift)/scale;
    const double a = 0.5*(zmax - zmin);
    const double b = 0.5*(zmin + zmax);

    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        m_nodes[i] = a*m_glq_nodes[i] + b;

    m_affine_legendre.init([&](std::span<double> x){
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            x[i] = shift + scale*m_nodes[i];
    });
    m_affine_legendre.iterate(integrals.order() - 1);

    const std::size_t last_extent
        = integrals.order() + integrals.extra_extent();
    zest::MDSpan<double, 2> legendre(
            m_legendre.data(), {last_extent, m_glq_weights.size()});
    legendre_recursion_vec(legendre, m_nodes);

    for (std::size_t l = 0; l < last_extent; ++l)
    {
        double res = 0.0;
        for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
            res += m_glq_weights[i]*m_affine_legendre.current()[i]*legendre(l, i);
        integrals(integrals.order() - 1, l) = a*res;
    }

    if constexpr (max_forward_order < 1)
        if (integrals.order() == 1) return;

    for (std::size_t l = 0; l < last_extent - 1; ++l)
    {
        double res = 0.0;
        for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
            res += m_glq_weights[i]*m_affine_legendre.prev()[i]*legendre(l, i);
        integrals(integrals.order() - 2, l) = a*res;
    }

    for (std::size_t n = integrals.order() - 1; n >= max_forward_order + 2; --n)
    {
        for (std::size_t l = 0; l < n + integrals.extra_extent() - 1; ++l)
        {
            integrals(n - 2, l)
                = (double(2*n - 1)/double(n - 1))*shift*integrals(n - 1, l)
                    - (double(n)/double(n - 1))*integrals(n, l)
                    + (double(2*n - 1)/double(n - 1))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                    + (double(2*n - 1)/double(n - 1))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
        }
    }
}

// Operates in the region where `1 - shift < scale < 1 + shift`
void AffineLegendreIntegralRecursion::integrals_partial_interval(
    TrapezoidSpan<double> integrals, double shift, double scale)
{
    /*
        As a compromise between computing all integrals via expensive direct numerical integration, and computing all integrals via the fast but unstable recursion formula, the integrals here are computed by first computing two seed rows via numerical integration, and then expanding outward using backward and forward recursion. The number of steps to compute via recursion can be chosen arbitrarily. Smaller numbers mitigate error arising from the unstable recursion, while larger numbers increase performance because less integrals need to be computed numerically.

        As a special case, the `l == 0` and `n == 0` reduce to simple integrals of Legendre polynomials, which this algorithm takes advantage of.

        On the diagonal boundary the forward recursion would require elements from outside the computed range, so direct integration must be used.

        Schematic view of the algorithm:
            LLLL
            LFFII
            LFFFII

            LBBBBBB
            LBBBBBBB
            LIIIIIIII
            LIIIIIIIII
            LFFFFFFFFII
            LFFFFFFFFFII
            ...
            LBBBBBBBBBBBB
            LBBBBBBBBBBBBB
            LIIIIIIIIIIIIII
            LIIIIIIIIIIIIIII
            LFFFFFFFFFFFFFFII
            LFFFFFFFFFFFFFFFII

            LBBBBBBBBBBBBBBBBBB
            LIIIIIIIIIIIIIIIIIII
            LIIIIIIIIIIIIIIIIIIII
        Here:
            L - integals computed via Legendre integral recursion
            F - integrals computed via forward recursion
            B - integrals computed via backward recursion
            I - integrals computed via direct numerical integration
    */

    assert(1.0 < shift + scale);
    assert(scale < shift + 1);
    if (integrals.order() == 0) return;

    const double zmin = std::max(-1.0, -(1.0 + shift)/scale);
    const double zmax = (1.0 - shift)/scale;
    const double half_width = 0.5*(zmax - zmin);
    const double mid_point = 0.5*(zmin + zmax);

    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        m_nodes[i] = half_width*m_glq_nodes[i] + mid_point;

    const std::size_t last_extent
        = integrals.order() + integrals.extra_extent();
    zest::MDSpan<double, 2> weighted_legendre(
            m_legendre.data(), {last_extent, m_glq_weights.size()});
    legendre_recursion_vec(weighted_legendre, m_nodes);
    for (std::size_t l = 0; l < weighted_legendre.extents()[0]; ++l)
    {
        for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
            weighted_legendre(l, i) *= m_glq_weights[i];
    }

    m_affine_legendre.init([&](std::span<double> x){
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            x[i] = shift + scale*m_nodes[i];
    });

    // Integrals for `n == 0` row.
    m_leg_int_rec.legendre_integral(m_leg_int_top, (1.0 - shift)/scale);
    for (std::size_t l = 0; l < integrals.extra_extent() + 1; ++l)
        integrals(0, l) = m_leg_int_top[l];
    
    if (integrals.order() == 1) return;

    // Integrals for `l == 0`.
    m_leg_int_rec.legendre_integral(m_leg_int_bot, shift - scale);

    const double inv_scale = 1.0/scale;

    // Integrals for `n == 1` row.
    std::span<const double> affine_legendre = m_affine_legendre.next();
    first_step(
            shift, scale, half_width, inv_scale, affine_legendre, weighted_legendre, integrals);

    constexpr std::size_t max_recursion = 3;

    // Do remainder of the first half-block of rows.
    const std::size_t klim = std::min(max_recursion, integrals.order() - 1);
    for (std::size_t k = 1; k < klim; ++k)
    {
        std::span<const double> affine_legendre = m_affine_legendre.next();
        forward_recursion_step(
                shift, scale, half_width, inv_scale, 1 + k, affine_legendre, weighted_legendre, integrals);
    }
    assert(1 + klim <= integrals.order());
    if (1 + klim == integrals.order()) return;

    const std::size_t block_size = 2*(max_recursion + 1);
    const std::size_t num_blocks
        = (integrals.order() - (max_recursion + 1))/block_size;
    
    // Do full blocks.
    for (std::size_t block_index = 0; block_index < num_blocks; ++block_index)
    {
        const std::size_t nmid = (1 + block_index)*block_size;
        m_affine_legendre.iterate(max_recursion + 2);
        glq_step(
                half_width, inv_scale, nmid - 1, m_affine_legendre.prev(), weighted_legendre, integrals);
        glq_step(
                half_width, inv_scale, nmid, m_affine_legendre.current(), weighted_legendre, integrals);
        for (std::size_t k = 0; k < max_recursion; ++k)
            backward_recursion_step(
                    shift, scale, inv_scale, nmid - (k + 2), integrals);
        for (std::size_t k = 0; k < max_recursion; ++k)
        {
            std::span<const double> affine_legendre = m_affine_legendre.next();
            forward_recursion_step(
                    shift, scale, half_width, inv_scale, nmid + k + 1, affine_legendre, weighted_legendre, integrals);
        }
    }

    const std::size_t num_remaining
        = integrals.order() - (max_recursion + 1) - num_blocks*block_size;
    

    if (num_remaining == 0) return;
    
    // If only one row remains it is computed via numerical integration.
    if (num_remaining == 1)
    {
        const std::size_t nmid = integrals.order() - 1;
        std::span<const double> affine_legendre = m_affine_legendre.next();
        glq_step(half_width, inv_scale, nmid, affine_legendre, weighted_legendre, integrals);
    }
    // If the number of remaining rows is not enough to jump to the middle of the next block, backward recursion from the last two rows is used.
    else if (num_remaining < max_recursion + 2)
    {
        const std::size_t nmid = integrals.order() - 1;
        m_affine_legendre.iterate(num_remaining);
        glq_step(
                half_width, inv_scale, nmid, m_affine_legendre.current(), weighted_legendre, integrals);
        glq_step(
                half_width, inv_scale, nmid - 1, m_affine_legendre.prev(), weighted_legendre, integrals);
        const std::size_t klim = num_remaining - 2;
        for (std::size_t k = 0; k < klim; ++k)
            backward_recursion_step(
                    shift, scale, inv_scale, nmid - (k + 2), integrals);
    }
    // If possible, the remaining incomplete block is computed similarly as full blocks.
    else
    {
        const std::size_t nmid = (1 + num_blocks)*block_size;
        m_affine_legendre.iterate(max_recursion + 2);
        glq_step(
                half_width, inv_scale, nmid - 1, m_affine_legendre.prev(), weighted_legendre, integrals);
        glq_step(
                half_width, inv_scale, nmid, m_affine_legendre.current(), weighted_legendre, integrals);
        for (std::size_t k = 0; k < max_recursion; ++k)
            backward_recursion_step(
                    shift, scale, inv_scale, nmid - (k + 2), integrals);
        for (std::size_t k = 0; k < max_recursion - (block_size - num_remaining); ++k)
        {
            auto affine_legendre = m_affine_legendre.next();
            forward_recursion_step(
                    shift, scale, half_width, inv_scale, nmid + k + 1, affine_legendre, weighted_legendre, integrals);
        }
    }
}

void AffineLegendreIntegralRecursion::first_step(
    double shift, double scale, double half_width, double inv_scale, std::span<const double> affine_legendre, zest::MDSpan<double, 2> weighted_legendre, TrapezoidSpan<double> integrals)
{
    const std::size_t lmax = integrals.extra_extent() + 1;
    integrals(1, 0) = -inv_scale*m_leg_int_bot[1];
    for (std::size_t l = 1; l < lmax - 1; ++l)
    {
        integrals(1, l) = shift*integrals(0, l)
                + (double(l + 1)/double(2*l + 1))*scale*integrals(0, l + 1)
                + (double(l)/double(2*l + 1))*scale*integrals(0, l - 1);
    }
    double res = 0.0;
    for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
        res += affine_legendre[i]*weighted_legendre(lmax - 1, i);
    integrals(1, lmax - 1) = half_width*res;
    res = 0.0;
    for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
        res += affine_legendre[i]*weighted_legendre(lmax, i);
    integrals(1, lmax) = half_width*res;
}

void AffineLegendreIntegralRecursion::glq_step(
    double half_width, double inv_scale, std::size_t n, std::span<const double> affine_legendre, zest::MDSpan<double, 2> weighted_legendre, TrapezoidSpan<double> integrals)
{
    const std::size_t extent = integrals.extra_extent() + n + 1;
    integrals(n, 0) = -inv_scale*m_leg_int_bot[n];
    for (std::size_t l = 1; l < extent; ++l)
    {
        double res = 0.0;
        for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
            res += affine_legendre[i]*weighted_legendre(l, i);
        integrals(n, l) = half_width*res;
    }
}

void AffineLegendreIntegralRecursion::forward_recursion_step(
    double shift, double scale, double half_width, double inv_scale, std::size_t n, std::span<const double> affine_legendre, zest::MDSpan<double, 2> weighted_legendre, TrapezoidSpan<double> integrals)
{
    const std::size_t lmax = integrals.extra_extent() + n;
    integrals(n, 0) = -inv_scale*m_leg_int_bot[n];
    for (std::size_t l = 1; l < lmax - 1; ++l)
    {
        integrals(n, l) = (double(2*n - 1)/double(n))*shift*integrals(n - 1, l)
                - (double(n - 1)/double(n))*integrals(n - 2, l)
                + (double(2*n - 1)/double(n))*(double(l + 1)/double(2*l + 1))*scale*integrals(n - 1, l + 1)
                + (double(2*n - 1)/double(n))*(double(l)/double(2*l + 1))*scale*integrals(n - 1, l - 1);
    }
    double res = 0.0;
    for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
        res += affine_legendre[i]*weighted_legendre(lmax - 1, i);
    integrals(n, lmax - 1) = half_width*res;
    res = 0.0;
    for (std::size_t i = 0; i < m_glq_weights.size(); ++i)
        res += affine_legendre[i]*weighted_legendre(lmax, i);
    integrals(n, lmax) = half_width*res;
}

void AffineLegendreIntegralRecursion::backward_recursion_step(
    double shift, double scale, double inv_scale, std::size_t n, TrapezoidSpan<double> integrals)
{
    const std::size_t extent = integrals.extra_extent() + n + 1;
    integrals(n, 0) = -inv_scale*m_leg_int_bot[n];
    for (std::size_t l = 1; l < extent; ++l)
    {
        integrals(n, l)
            = (double(2*n + 3)/double(n + 1))*shift*integrals(n + 1, l)
                - (double(n + 2)/double(n + 1))*integrals(n + 2, l)
                + (double(2*n + 3)/double(n + 1))*(double(l + 1)/double(2*l + 1))*scale*integrals(n + 1, l + 1)
                + (double(2*n + 3)/double(n + 1))*(double(l)/double(2*l + 1))*scale*integrals(n + 1, l - 1);
    }
}