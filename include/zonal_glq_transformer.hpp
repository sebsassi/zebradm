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

#include <span>
#include <vector>

#include <zest/gauss_legendre.hpp>
#include <zest/md_span.hpp>
#include <zest/real_sh_expansion.hpp>
#include <zest/sh_conventions.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "legendre.hpp"

namespace zdm::zebra
{

/**
    @brief Fast transformations from a Gauss-Legendre quadrature grid to zonal spherical harmonic components.

    @tparam NORM normalization convention of spherical harmonics
    @tparam GridLayoutType
*/
template <zest::st::SHNorm sh_norm_param, typename GridLayoutType = zest::st::DefaultLayout>
class ZonalGLQTransformer
{
public:
    using GridLayout = GridLayoutType;

    static constexpr zest::st::SHNorm sh_norm = sh_norm_param;

    ZonalGLQTransformer() = default;
    explicit ZonalGLQTransformer(std::size_t order):
        m_glq_nodes(zest::gl::PackedLayout::size(GridLayout::lat_size(order))),
        m_glq_weights(zest::gl::PackedLayout::size(GridLayout::lat_size(order))), m_leg_grid(order*GridLayout::lat_size(order)), m_longitudinal_average(GridLayout::lat_size(order)), m_order(order)
    {
        zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double z = m_glq_nodes[i];
            std::span leg = std::span(m_leg_grid.data() + i*order, order);
            legendre_recursion(leg, z);
            for (std::size_t l = 0; l < order; ++l)
            {
                if constexpr (sh_norm_param == zest::st::SHNorm::qm)
                    leg[l] *= 0.5*std::numbers::inv_sqrtpi*std::sqrt(double(2*l + 1));
                else
                    leg[l] *= std::sqrt(double(2*l + 1));
            }
        }
    }

    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_glq_nodes.resize(zest::gl::PackedLayout::size(GridLayout::lat_size(order)));
        m_glq_weights.resize(zest::gl::PackedLayout::size(GridLayout::lat_size(order)));
        m_leg_grid.resize(order*GridLayout::lat_size(order));
        m_longitudinal_average.resize(GridLayout::lat_size(order));
        zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double z = m_glq_nodes[i];
            std::span leg = std::span(m_leg_grid.data() + i*order, order);
            legendre_recursion(leg, z);
            for (std::size_t l = 0; l < order; ++l)
            {
                if constexpr (sh_norm_param == zest::st::SHNorm::qm)
                    leg[l] *= 0.5*std::numbers::inv_sqrtpi*std::sqrt(double(2*l + 1));
                else
                    leg[l] *= std::sqrt(double(2*l + 1));
            }
        }

        m_order = order;
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to zonal harmonic coefficients.

        @param values values on the spherical quadrature grid
        @param expansion coefficients of the expansion
    */
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
                zest::MDSpan<const double, 1> values_i = values[i];
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
                zest::MDSpan<const double, 1> values_i = values[i];
                for (std::size_t j = 0; j < values.shape()[lat_axis]; ++j)
                    m_longitudinal_average[j] += values_i[j];
            }
        }

        constexpr double sh_normalization = zest::st::normalization<sh_norm_param>();
        const double prefactor = sh_normalization*(2.0*std::numbers::pi)/double(values.shape()[lon_axis]);

        for (std::size_t i = 0; i < m_longitudinal_average.size(); ++i)
            m_longitudinal_average[i] *= prefactor;
        
        const std::size_t min_order = std::min(expansion.size(), values.order());

        const std::size_t num_lat = m_longitudinal_average.size();
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;
        const std::size_t lparity = min_order & 1;

        std::ranges::fill(expansion, 0.0);
        const double south = m_longitudinal_average[south_offset];
        const double north = m_longitudinal_average[north_offset];
        const double symmetry_factor = (num_lat & 1) ? 0.5 : 1.0;
        const std::array<double, 2> symm_asymm = {
            symmetry_factor*(north + south), symmetry_factor*(north - south)
        };
        const std::array<double, 2> weighted = {
            m_glq_weights[0]*symm_asymm[0], m_glq_weights[0]*symm_asymm[1]
        };
        const std::array<double, 2> weighted_l = {
            weighted[lparity], weighted[lparity ^ 1]
        };
        std::span leg(m_leg_grid.data(), m_order);
        expansion[0] += double(lparity)*weighted[0]*leg[0];
        for (std::size_t l = lparity; l < min_order; l += 2)
        {
            expansion[l] += weighted_l[0]*leg[l];
            expansion[l + 1] += weighted_l[1]*leg[l + 1];
        }

        for (std::size_t i = 1; i < num_unique_nodes; ++i)
        {
            const double south = m_longitudinal_average[south_offset - i];
            const double north = m_longitudinal_average[north_offset + i];
            const std::array<double, 2> symm_asymm = {
                north + south, north - south
            };
            const std::array<double, 2> weighted = {
                m_glq_weights[i]*symm_asymm[0], m_glq_weights[i]*symm_asymm[1]
            };
            const std::array<double, 2> weighted_l = {
                weighted[lparity], weighted[lparity ^ 1]
            };
            std::span leg(m_leg_grid.data() + i*m_order, m_order);
            expansion[0] += double(lparity)*weighted[0]*leg[0];
            for (std::size_t l = lparity; l < min_order; l += 2)
            {
                expansion[l] += weighted_l[0]*leg[l];
                expansion[l + 1] += weighted_l[1]*leg[l + 1];
            }
        }
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to zonal harmonic coefficients.

        @param values values on the spherical quadrature grid
    */
    std::vector<double> forward_transform(
        zest::st::SphereGLQGridSpan<const double, GridLayout> values, 
        std::size_t order)
    {
        std::vector<double> res(order);
        forward_transform(values, res);
        return res;
    }

private:
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_leg_grid;
    std::vector<double> m_longitudinal_average;
    std::size_t m_order;
};

} // namespace zdm::zebra
