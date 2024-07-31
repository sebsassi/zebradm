#include "zebra_angle_integrator.hpp"

#include "coordinates/coordinate_functions.hpp"

#include "radon_util.hpp"

namespace zebra
{

IsotropicAngleIntegrator::IsotropicAngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 2),
    m_rotor(dist_order + 2),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_integrator(dist_order + 2),
    m_dist_order(dist_order) {}

void IsotropicAngleIntegrator::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(geg_order);
    m_rotor.expand(geg_order);
    m_geg_zernike_exp.resize(geg_order);
    m_rotated_geg_zernike_exp.resize(geg_order);
    m_integrator.resize(geg_order);
    m_dist_order = dist_order;
}

void IsotropicAngleIntegrator::integrate(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out)
{
    resize(distribution.order());
    util::apply_gegenbauer_reduction(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < boosts.size(); ++i)
        integrate(boosts[i], min_speeds, out[i]);
}

void IsotropicAngleIntegrator::integrate(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<double> out)
{
    resize(distribution.order());
    util::apply_gegenbauer_reduction(distribution, m_geg_zernike_exp);
    integrate(boost, min_speeds, out);
}

void IsotropicAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds,
    std::span<double> out)
{
    const double boost_speed = length(boost);
    const double boost_colat = std::acos(boost[2]/boost_speed);
    const double boost_az = std::atan2(boost[1], boost[0]);

    const Vector<double, 3> euler_angles = {
        0.0, -boost_colat, std::numbers::pi - boost_az
    };

    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    for (std::size_t n = 0; n < m_geg_zernike_exp.order(); ++n)
        m_rotor.rotate(
                m_rotated_geg_zernike_exp[n], m_wigner_d_pi2, euler_angles);

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
    {
        out[i] = m_integrator.integrate(
                m_rotated_geg_zernike_exp, boost_speed, min_speeds[i]);
    }
}

AnisotropicAngleIntegrator::AnisotropicAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 2, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_grids((dist_order + 2)*zest::st::SphereGLQGridSpan<double>::size(std::min(dist_order + 2 + resp_order, trunc_order))),
    m_integrator(dist_order + 2, resp_order, std::min(dist_order + 2 + resp_order, trunc_order)) {}

void AnisotropicAngleIntegrator::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 2, resp_order));
    
    if (dist_order != m_dist_order)
    {
        m_geg_zernike_exp.resize(dist_order + 2);
        m_rotated_geg_zernike_exp.resize(dist_order + 2);
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                (dist_order + 2)*zest::st::SphereGLQGridSpan<double>::size(std::min(dist_order + 2 + resp_order, trunc_order)));
        m_integrator.resize(
                dist_order + 2, resp_order, std::min(dist_order + 2 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AnisotropicAngleIntegrator::integrate(
    DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, 
    std::span<const double> min_speeds, ResponseSpan response,
    std::span<const double> era, zest::MDSpan<double, 2> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    util::apply_gegenbauer_reduction(distribution, m_geg_zernike_exp);

    for (std::size_t i = 0; i < boosts.size(); ++i)
        integrate(
                boosts[i], min_speeds, response, era[i], geg_order, top_order, 
                out[i]);
}

void AnisotropicAngleIntegrator::integrate(
    DistributionSpan distribution, const Vector<double, 3>& boost, 
    std::span<const double> min_speeds, ResponseSpan response, double era, 
    zest::MDSpan<double, 2> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    util::apply_gegenbauer_reduction(distribution, m_geg_zernike_exp);

    integrate(boost, min_speeds, response, era, geg_order, top_order, out);
}

void AnisotropicAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds, 
    ResponseSpan response, double era, std::size_t geg_order,
    std::size_t top_order, std::span<double> out)
{
    using ZernikeSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>;

    const auto& [boost_speed, boost_colat, boost_az]
        = coordinates::cartesian_to_spherical_phys(boost);

    const Vector<double, 3> euler_angles = {
        0.0, -boost_colat, std::numbers::pi - boost_az
    };

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), {geg_order}, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        ZernikeSpan::SubSpan rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        std::ranges::copy(m_geg_zernike_exp[n].flatten(), m_rotated_geg_zernike_exp.begin());
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
        out[i] = m_integrator.integrate(
                rotated_geg_zernike_grids, response[i], era, boost, 
                min_speeds[i], m_wigner_d_pi2);
}

} // namespace zebra