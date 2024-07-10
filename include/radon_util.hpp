#pragma once

#include "radon_transformer.hpp"

#include <cassert>

#define RESTRICT(P) P __restrict__

namespace detail
{

// multiply b to a: `a *= b`
constexpr void mul(RESTRICT(double*) a, const double* b, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] *= b[i];
}

// multiply b to a: `a *= b`
constexpr void mul(std::span<double> a, std::span<const double> b)
{
    const std::size_t size = std::min(a.size(), b.size());
    mul(a.data(), b.data(), size);
}

// multiply c and b and add to a: `a += b*c`
constexpr void fmadd(
    RESTRICT(double*) a, const double* b, const double* c, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i)
        a[i] += b[i]*c[i];
}

// multiply c and b and add to a: `a += b*c`
constexpr void fmadd(
    std::span<double> a, std::span<const double> b, std::span<const double> c)
{
    assert(b.size() == c.size());
    const std::size_t size = std::min(a.size(), b.size());
    fmadd(a.data(), b.data(), c.data(), size);
}

template <typename FuncType>
void evaluate_legendre_expansion_grid(
    LegendreArrayRecursion& leg_recursion, zest::MDSpan<const double, 2> coeffs, FuncType&& grid_point_generator, std::span<double> out)
{
    assert(leg_recursion.size() == coeffs.extents().back());
    assert(leg_recursion.size() == out.size());
    leg_recursion.init(grid_point_generator);
    std::ranges::fill(out, 0.0);
    for (std::size_t l = 0; l < coeffs.extents().front(); ++l)
        fmadd(out, coeffs[l], leg_recursion.next());
}

void apply_legendre_shift(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> expansion, 
    zest::TriangleSpan<const double, zest::TriangleLayout> shifted_leg_coeff, 
    SHExpansionCollectionSpan<std::array<double, 2>> out);

template <zest::zt::ZernikeNorm NORM>
constexpr double geg_rec_coeff(std::size_t n)
{
    if constexpr (NORM == zest::zt::ZernikeNorm::UNNORMED)
        return 1.0/double(2*n + 3);
    else
        return 1.0/std::sqrt(double(2*n + 3));
}

void apply_gegenbauer_recursion(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out);

template <zest::st::SHNorm SH_NORM, typename GridLayoutType = zest::st::DefaultLayout>
class ZonalGLQTransformer
{
public:
    using GridLayout = GridLayoutType;

    static constexpr SHNorm sh_norm = SH_NORM;

    ZonalGLQTransformer() = default;
    explicit ZonalGLQTransformer(std::size_t order):
        m_glq_nodes(zest::gl::PackedLayout::size(GridLayout::lat_size(order))),
        m_glq_weights(zest::gl::PackedLayout::size(GridLayout::lat_size(order))), m_leg_grid(order*GridLayout::lat_size(order)), m_order(order)
    {
        zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double z = m_glq_nodes[i];
            legendre_recursion(std::span(m_leg_grid.data() + i*order, order), z);
        }
    }

    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_glq_nodes.resize(zest::gl::PackedLayout::size(GridLayout::lat_size(order)));
        m_glq_weights.resize(zest::gl::PackedLayout::size(GridLayout::lat_size(order)));
        zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double z = m_glq_nodes[i];
            legendre_recursion(std::span(m_leg_grid.data() + i*order, order), z);
        }
    }

    void forward_transform(
        zest::st::SphereGLQGridSpan<const double, GridLayout> values, std::span<double> expansion)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;

        resize(values.order());

        if constexpr (std::same_as<GridLayout, zest::st::LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < values.shape()[lat_axis]; ++i)
            {
                zest::MDSpan<double, 1> values_i = values[i];
                m_longitudinal_average[i] = 0.0;
                for (std::size_t j = 0; j < values.shape()[lon_axis]; ++j)
                    m_longitudinal_average[i] += values_i[j];
            }
        }
        else if constexpr (std::same_as<GridLayout, zest::st::LonLatLayout<typename GridLayout::Alignment>>)
        {
            std::ranges::fill(m_longitudinal_average, 0.0);
            for (std::size_t i = 0; i < values.shape()[lon_axis]; ++i)
            {
                zest::MDSpan<double, 1> values_i = values[i];
                for (std::size_t j = 0; j < values.shape()[lat_axis]; ++j)
                    m_longitudinal_average[j] += values_i[j];
            }
        }

        constexpr double sh_normalization = normalization<SH_NORM>();
        const double prefactor = sh_normalization*(2.0*std::numbers::pi)/double(values.shape()[lon_axis]);

        for (std::size_t i = 0; i < m_longitudinal_average.size(); ++i)
            m_longitudinal_average[i] *= prefactor;
        
        const std::size_t min_order = std::min(expansion.size(), m_longitudinal_average.size());

        std::ranges::fill(expansion, 0.0);
        for (std::size_t i = 0; i < m_longitudinal_average.size(); ++i)
        {
            for (std::size_t l = 0; l < min_order; ++l)
            {
                expansion[l] += m_glq_weights[i]*m_leg_grid(i, l);
            }
        }
    }

private:
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_leg_grid;
    std::vector<double> m_longitudinal_average;
    std::size_t m_order;
};

}