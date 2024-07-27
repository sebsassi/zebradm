#include "radon_transformer.hpp"

#include "coordinates/coordinate_functions.hpp"

using TriangleStackSpan = SuperSpan<zest::TriangleSpan<double, zest::TriangleLayout>>;

using SphereGridStackSpan = SuperSpan<zest::st::SphereGLQGrid<double>>;

void RadonTransformer::angle_integrated_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution,
    std::span<const Vector<double, 3>> boosts,
    std::span<const double> min_speeds, zest::MDSpan<double, 2> out)
{
    RadonTransformer::angle_integrated_transform(
            distribution, boosts, min_speeds,
            std::numeric_limits<std::size_t>::max(), out);
}

void RadonTransformer::angle_integrated_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution,
    std::span<const Vector<double, 3>> boosts,
    std::span<const double> min_speeds, std::size_t trunc_order, 
    zest::MDSpan<double, 2> out)
{
    const std::size_t dist_order = distribution.order();
    constexpr std::size_t resp_order = 0;
    resize(dist_order, resp_order, trunc_order);
    const auto& [geg_order, top_order]
        = geg_top_orders(dist_order, resp_order, trunc_order);

    detail::apply_gegenbauer_recursion(distribution, m_geg_zernike_exp);

    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        const double boost_speed = length(boosts[i]);
        const double boost_colat = std::acos(boosts[i][2]/boost_speed);
        const double boost_az = std::atan2(boosts[i][1], boosts[i][0]);

        const Vector<double, 3> euler_angles = {
            0.0, -boost_colat, std::numbers::pi - boost_az
        };

        std::ranges::copy(
            m_geg_zernike_exp.flatten(),
            m_rotated_geg_zernike_exp.flatten().begin());

        for (std::size_t n = 0; n < geg_order; ++n)
            m_rotor.rotate(m_rotated_geg_zernike_exp[n], euler_angles);

        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            if (min_speeds[j] > 1.0 + boost_speed)
            {
                out(i, j) = 0.0;
                continue;
            }

            TrapezoidSpan<double> aff_leg_ylm_integrals
                = evaluate_aff_leg_ylm_integrals(min_speeds[j], boost_speed, geg_order, resp_order);
            
            double res = 0;
            for (std::size_t n = 0; n < geg_order; ++n)
            {
                for (std::size_t l = n & 1; l <= n; l += 2)
                    res += m_rotated_geg_zernike_exp(n, l, 0)[0]*aff_leg_ylm_integrals(n, l);
            }

            out(i, j) = (2.0*std::numbers::pi)*res;
        }
    }
}



void RadonTransformer::angle_integrated_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution,
    std::span<const Vector<double, 3>> boosts,
    std::span<const double> min_speeds,
    SHExpansionCollectionSpan<const std::array<double, 2>> response, 
    std::span<const double> era, zest::MDSpan<double, 2> out)
{
    RadonTransformer::angle_integrated_transform(
            distribution, boosts, min_speeds, response, era, 
            std::numeric_limits<std::size_t>::max(), out);
}

void RadonTransformer::angle_integrated_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution,
    std::span<const Vector<double, 3>> boosts,
    std::span<const double> min_speeds,
    SHExpansionCollectionSpan<const std::array<double, 2>> response, 
    std::span<const double> era, std::size_t trunc_order,
    zest::MDSpan<double, 2> out)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const auto& [geg_order, top_order]
        = geg_top_orders(dist_order, resp_order, trunc_order);

    detail::apply_gegenbauer_recursion(distribution, m_geg_zernike_exp);

    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        const double boost_speed = length(boosts[i]);
        const double boost_colat = std::acos(boosts[i][2]/boost_speed);
        const double boost_az = std::atan2(boosts[i][1], boosts[i][0]);

        const Vector<double, 3> euler_angles = {
            0.0, -boost_colat, std::numbers::pi - boost_az
        };

        std::ranges::copy(
                m_geg_zernike_exp.flatten(),
                m_rotated_geg_zernike_exp.flatten().begin());

        SuperSpan<zest::st::SphereGLQGridSpan<double>>
        rotated_geg_zernike_grids(
                m_rotated_geg_zernike_grids.data(), {geg_order}, top_order);

        for (std::size_t n = 0; n < geg_order; ++n)
        {
            std::ranges::copy(m_geg_zernike_exp.flatten(), m_rotated_geg_zernike_exp.flatten().begin());
            m_rotor.rotate(m_rotated_geg_zernike_exp[0], euler_angles);
            m_glq_transformer.backward_transform(
                    m_rotated_geg_zernike_exp[0], rotated_geg_zernike_grids[n]);
        }

        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            if (min_speeds[j] > 1.0 + boost_speed)
            {
                out(i, j) = 0.0;
                continue;
            }

            const double boost_speed = length(boosts[i]);
            const double boost_colat = std::acos(boosts[i][2]/boost_speed);
            const double boost_az = std::atan2(boosts[i][1], boosts[i][0]);

            const Vector<double, 3> euler_angles = {
                0.0, -boost_colat, std::numbers::pi - boost_az - era[i]
            };

            std::ranges::copy(
                    response[j].flatten(), m_rotated_response_exp.flatten().begin());

            m_rotor.rotate(m_rotated_response_exp, euler_angles);

            m_glq_transformer.backward_transform(
                    m_rotated_response_exp, m_rotated_response_grid);

            TrapezoidSpan<double> aff_leg_ylm_integrals
                = evaluate_aff_leg_ylm_integrals(
                        min_speeds[j], boost_speed, geg_order, resp_order);

            const std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
            double res = 0;
            for (std::size_t n = 0; n < geg_order; ++n)
            {
                std::ranges::copy(
                        m_rotated_response_grid.flatten(),
                        m_rotated_grid.flatten().begin());
                detail::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
                m_zonal_transformer.forward_transform(
                        m_rotated_grid, m_rotated_exp);
                for (std::size_t l = 0; l <= n + extra_extent; ++l)
                    res += m_rotated_exp[l]*aff_leg_ylm_integrals(n, l);
            }

            out(i, j) = (2.0*std::numbers::pi)*res;
        }
    }
}

void RadonTransformer::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    const auto& [geg_order, top_order]
        = geg_top_orders(dist_order, resp_order, trunc_order);

    if (geg_order != m_geg_zernike_exp.order())
    {
        m_geg_zernike_exp.resize(geg_order);
        m_rotated_geg_zernike_exp.resize(geg_order);
    }
    
    if (resp_order != m_rotated_response_exp.order())
        m_rotated_response_exp.resize(resp_order);

    if (top_order != m_rotated_grid.order())
    {
        std::size_t top_sphere_grid_size
            = zest::st::SphereGLQGrid<double>::Layout::size(top_order);
        m_rotated_geg_zernike_grids.resize(geg_order*top_sphere_grid_size);
        m_rotated_response_grid.resize(top_order);
        m_rotated_grid.resize(top_order);
        m_rotated_exp.resize(top_order);

        std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
        m_aff_leg_ylm_integrals.resize(
                TrapezoidLayout::size(geg_order, extra_extent));

        m_zonal_transformer.resize(top_order);
        m_glq_transformer.resize(top_order);
        m_aff_leg_int_rec.resize(geg_order, extra_extent);
    }
}

TrapezoidSpan<double> RadonTransformer::evaluate_aff_leg_ylm_integrals(
    double min_speed, double boost_speed, std::size_t geg_order, std::size_t resp_order)
{
    const std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
    TrapezoidSpan<double> integrals(m_aff_leg_ylm_integrals.data(), geg_order, extra_extent);
    m_aff_leg_int_rec.integrals(integrals, min_speed, boost_speed);
    
    for (std::size_t n = 0; n < geg_order; ++n)
    {
        for (std::size_t l = 0; l <= n + extra_extent; ++l)
            integrals(n, l) *= (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
    }

    return integrals;
}