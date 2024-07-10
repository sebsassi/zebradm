#include "radon_transformer.hpp"

#include "radon_util.hpp"

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
    resize(dist_order, resp_order, trunc_order, min_speeds.size());
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

        std::ranges::copy(m_geg_zernike_exp.flatten(), m_rotated_geg_zernike_exp.flatten().begin());
        for (std::size_t n = 0; n < geg_order; ++n)
            m_rotor.rotate(m_rotated_geg_zernike_exp[n], euler_angles);

        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            if (min_speeds[j] > 1.0 + boost_speed)
            {
                out(i, j) = 0.0;
                continue;
            }

            zest::TriangleSpan<double, zest::TriangleLayout> 
            aff_leg_ylm_integrals
                = evaluate_aff_leg_ylm_integrals(min_speeds[j], boost_speed);

            double res = 0;
            for (std::size_t n = 0; n < geg_order; ++n)
            {
                for (std::size_t l = n & 1; l <= n; ++l)
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
    resize(dist_order, resp_order, trunc_order, min_speeds.size());
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

        std::ranges::copy(m_geg_zernike_exp.flatten(), m_rotated_geg_zernike_exp.flatten().begin());

        SuperSpan<zest::st::SphereGLQGrid<double>> rotated_geg_zernike_grids;
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
                0.0, -boost_colat, std::numbers::pi - boost_az - era
            };

            std::ranges::copy(response[j].flatten(), m_rotated_response_exp.flatten().begin());
            m_rotor.rotate(m_rotated_response_exp, euler_angles);

            m_glq_transformer.backward_transform(
                    m_rotated_response_exp, m_rotated_response_grid);

            zest::TriangleSpan<double, zest::TriangleLayout> 
            aff_leg_ylm_integrals
                = evaluate_aff_leg_ylm_integrals(min_speeds[j], boost_speed);

            double res = 0;
            for (std::size_t n = 0; n < geg_order; ++n)
            {
                std::ranges::copy(
                        m_rotated_response_grid.flatten(),
                        m_rotated_grid.flatten().begin());
                detail::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
                m_zonal_transformer.forward_transform(
                        m_rotated_grid, m_rotated_exp);
                for (std::size_t l = n & 1; l <= n; ++l)
                    res += m_rotated_exp[l]*aff_leg_ylm_integrals(n, l);
            }

            out(i, j) = (2.0*std::numbers::pi)*res;
        }
    }
}

void RadonTransformer::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order, 
    std::size_t min_speeds_size)
{
    const auto& [geg_order, top_order]
        = geg_top_orders(dist_order, resp_order, trunc_order);

    if (geg_order != m_geg_zernike_exp.order())
    {
        m_vmin_boost_sep_buffer.resize(
                geg_order*zest::TriangleLayout::size(geg_order));
        m_geg_zernike_exp.resize(geg_order);
        m_aff_legendre_buffer.resize(zest::TriangleLayout::size(geg_order));
        
        m_aff_leg_recursion.expand(geg_order);
    }
    
    if (resp_order != m_response_expansion.order())
        m_response_expansion.resize(resp_order);

    if (top_order != m_radon_grid.order())
    {
        std::size_t top_sphere_grid_size
            = zest::st::SphereGLQGrid<double>::Layout::size(top_order);
        m_geg_zernike_grids.resize(geg_order*top_sphere_grid_size);
        m_direction_grid.resize(top_sphere_grid_size);
        m_glq_transformer.resize(top_order);
        m_radon_grid.resize(top_order);
        m_rotated_response_grid.resize(top_order);
        m_radon_coeffs.resize(top_order);
        m_ylm_integrals.resize(top_order);
        m_lower_legendre_integrals.resize(top_order);
        m_grid_points.resize(top_order);
        
        m_leg_recursion.resize(top_sphere_grid_size);
        m_integral_recursion.expand(top_order);
    }

    m_vmin_part_grid_buffer.resize(
            min_speeds_size*geg_order
            *zest::st::SphereGLQGrid<double>::Layout::size(top_order));
}

MultiSuperSpan<zest::st::SphereGLQGridSpan<const double>, 2> 
RadonTransformer::vmin_contribution_grids(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> geg_zernike_exp,
    std::span<const double> min_speeds, std::size_t geg_order,
    std::size_t top_order)
{
    zest::TriangleSpan<double, zest::TriangleLayout> shifted_coeffs(
            m_aff_legendre_buffer, geg_order);

    MultiSuperSpan<zest::st::SphereGLQGridSpan<double>, 2> vmin_part_grids(
            m_vmin_part_grid_buffer, {min_speeds.size(), geg_order}, top_order);
    SHExpansionCollectionSpan<std::array<double, 2>> vmin_part_exps(
            m_vmin_boost_sep_buffer, {geg_order}, geg_order);
    
    for (std::size_t i = 0; i < min_speeds.size(); ++i)
    {
        m_aff_leg_recursion.evaluate_shifted(shifted_coeffs, min_speeds[i]);
        detail::apply_legendre_shift(
                geg_zernike_exp, shifted_coeffs, vmin_part_exps);
        for (std::size_t l = 0; l < geg_order; ++l)
            m_glq_transformer.backward_transform(
                    vmin_part_exps[l], vmin_part_grids(i,l));
    }

    return MultiSuperSpan<zest::st::SphereGLQGridSpan<const double>, 2>(m_vmin_part_grid_buffer, {min_speeds.size(), geg_order}, top_order);
}

zest::st::SphereGLQGridSpan<double> 
RadonTransformer::combine_vmin_boost_contributions(
    SuperSpan<zest::st::SphereGLQGridSpan<const double>> vmin_cont, 
    const Vector<double, 3>& boost, std::size_t top_order)
{
    auto boost_proj = [&](double lon, double colat) -> double
    {
        const Vector<double, 3> dir
                = coordinates::spherical_to_cartesian_phys(lon, colat);
        return dot(boost, dir);
    };
    
    std::span<double> flat_grid = m_radon_grid.flatten();
    auto boost_proj_gen = [&](std::span<double> span)
    {   
        assert(span.size() == flat_grid.size());
        zest::st::SphereGLQGridSpan<double> boost_proj_grid(span, top_order);
        m_grid_points.generate_values(boost_proj_grid, boost_proj);
    };

    zest::MDSpan<const double, 2> vmin_cont_flat(
        vmin_cont.flatten().data(), {vmin_cont.extents()[0], vmin_cont.subspan_size()});

    detail::evaluate_legendre_expansion_grid(
            m_leg_recursion, vmin_cont_flat, boost_proj_gen, flat_grid);

    return m_radon_grid;
}

double RadonTransformer::angle_integral(
    SHExpansionSpan<std::array<double, 2>> radon_coeffs,
    double min_speed, double boost_speed, const Vector<double, 3>& boost)
{
    const double zmin = -(1.0 + min_speed)/boost_speed;
    const double zmax = (1.0 - min_speed)/boost_speed;

    if (-1.0 >= zmax) return 0.0;

    // Only the l = 0 spherical harmonic has nonzero integral over the whole sphere.
    if (zmax >= 1.0 && -1.0 >= zmin)
        return 4.0*std::numbers::pi*radon_coeffs(0,0)[0];

    const double zmin_true = std::max(zmin, -1.0);
    const double zmax_true = std::min(zmax, 1.0);

    const double boost_colat = std::acos(boost[2]/boost_speed);
    const double boost_az = std::atan2(boost[1], boost[0]);

    const Vector<double, 3> euler_angles = {
        0.0, -boost_colat, std::numbers::pi - boost_az
    };

    m_rotor.rotate(radon_coeffs, euler_angles);
    evaluate_ylm_integrals(zmin_true, zmax_true);

    double res = 2.0*std::numbers::pi*(zmax_true - zmin_true)*radon_coeffs(0,0)[0];
    for (std::size_t l = 1; l < radon_coeffs.order(); ++l)
        res += radon_coeffs(l, 0)[0]*m_ylm_integrals[l];
    
    return res;
}

void RadonTransformer::evaluate_ylm_integrals(
    double lower_limit, double upper_limit)
{
    m_integral_recursion.legendre_integral(m_ylm_integrals, upper_limit);
    if (lower_limit > -1.0)
    {
        m_integral_recursion.legendre_integral(
                m_lower_legendre_integrals, lower_limit);
        for (std::size_t l = 0; l < m_ylm_integrals.size(); ++l)
            m_ylm_integrals[l] -= m_lower_legendre_integrals[l];
    }
    for (std::size_t l = 0; l < m_ylm_integrals.size(); ++l)
        m_ylm_integrals[l] *= 2.0*std::numbers::pi*std::sqrt(double(2*l + 1));

}