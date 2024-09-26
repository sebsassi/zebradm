#include "zest/zernike_expansion.hpp"
#include "zest/real_sh_expansion.hpp"
#include "zest/sh_glq_transformer.hpp"

namespace zebra
{

template <zest::st::SHNorm SH_NORM, typename GridLayoutType = zest::st::DefaultLayout>
class ZonalGLQTransformer
{
public:
    using GridLayout = GridLayoutType;

    static constexpr zest::st::SHNorm sh_norm = SH_NORM;

    ZonalGLQTransformer() = default;
    explicit ZonalGLQTransformer(std::size_t order):
        m_glq_nodes(zest::gl::PackedLayout::size(GridLayout::lat_size(order))),
        m_glq_weights(zest::gl::PackedLayout::size(GridLayout::lat_size(order))), m_leg_grid(order*GridLayout::lat_size(order)), m_longitudinal_average(GridLayout::lat_size(order)), m_order(order)
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
        m_leg_grid.resize(order*GridLayout::lat_size(order));
        m_longitudinal_average.resize(GridLayout::lat_size(order));
        zest::gl::gl_nodes_and_weights<zest::gl::PackedLayout, zest::gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double z = m_glq_nodes[i];
            legendre_recursion(std::span(m_leg_grid.data() + i*order, order), z);
        }

        m_order = order;
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
                zest::MDSpan<const double, 1> values_i = values[i];
                for (std::size_t j = 0; j < values.shape()[lat_axis]; ++j)
                    m_longitudinal_average[j] += values_i[j];
            }
        }

        constexpr double sh_normalization = zest::st::normalization<SH_NORM>();
        const double prefactor = sh_normalization*(2.0*std::numbers::pi)/double(values.shape()[lon_axis]);

        for (std::size_t i = 0; i < m_longitudinal_average.size(); ++i)
            m_longitudinal_average[i] *= prefactor;
        
        const std::size_t min_order = std::min(expansion.size(), m_longitudinal_average.size());

        std::ranges::fill(expansion, 0.0);
        for (std::size_t i = 0; i < m_longitudinal_average.size(); ++i)
        {
            std::span leg(m_leg_grid.data() + i*m_order, m_order);
            for (std::size_t l = 0; l < min_order; ++l)
                expansion[l] += m_glq_weights[i]*leg[l];
        }
    }

private:
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_leg_grid;
    std::vector<double> m_longitudinal_average;
    std::size_t m_order;
};

} // namespace zebra