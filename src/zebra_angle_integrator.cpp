#include "zebra_angle_integrator.hpp"

#include "coordinates/coordinate_functions.hpp"

#include "zebra_radon.hpp"
#include "radon_util.hpp"

namespace zebra
{

IsotropicAngleIntegrator::IsotropicAngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 2),
    m_rotor(dist_order + 2),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_integrator_core(dist_order + 2),
    m_dist_order(dist_order) {}

void IsotropicAngleIntegrator::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(geg_order);
    m_rotor.expand(geg_order);
    m_geg_zernike_exp.resize(geg_order);
    m_rotated_geg_zernike_exp.resize(geg_order);
    m_integrator_core.resize(geg_order);
    m_dist_order = dist_order;
}

void IsotropicAngleIntegrator::integrate(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < boosts.size(); ++i)
        integrate(boosts[i], min_speeds, out[i]);
}

void IsotropicAngleIntegrator::integrate(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<double> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    integrate(boost, min_speeds, out);
}

/**
    @brief Set Euler angles for rotation to a coordinate system whose z-axis is in the direction given by the arguments.

    @note The convention for the Euler angles is that used by `zest::Rotor::rotate`.
*/
template <zest::RotationType TYPE>
constexpr Vector<double, 3> euler_angles_to_align_z(
    double azimuth, double colatitude)
{
    if constexpr (TYPE == zest::RotationType::COORDINATE)
        return {azimuth, colatitude, 0.0};
    else
        return {0.0, -colatitude, std::numbers::pi - azimuth};
}

void IsotropicAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds,
    std::span<double> out)
{
    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    constexpr zest::RotationType rotation_type = zest::RotationType::COORDINATE;
    const auto& [boost_az, boost_colat, boost_speed]
        = coordinates::cartesian_to_spherical_phys(boost);
    const Vector<double, 3> euler_angles
        = euler_angles_to_align_z<rotation_type>(boost_az, boost_colat);
    
    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_rotated_geg_zernike_exp, boost_speed, min_speeds[i]);
}

[[nodiscard]] constexpr std::size_t geg_zernike_grids_size(
    std::size_t geg_order, std::size_t resp_order, std::size_t trunc_order
) noexcept
{
    return geg_order*zest::st::SphereGLQGridSpan<double>::size(std::min(geg_order + resp_order, trunc_order));
}

AnisotropicAngleIntegrator::AnisotropicAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 2, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(dist_order + 2, resp_order, trunc_order)),
    m_integrator_core(
        dist_order + 2, resp_order,
        std::min(dist_order + 2 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

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
                geg_zernike_grids_size(
                        dist_order + 2, resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 2, resp_order,
                std::min(dist_order + 2 + resp_order, trunc_order));
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

    zebra::radon_transform(distribution, m_geg_zernike_exp);

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

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    integrate(boost, min_speeds, response, era, geg_order, top_order, out);
}

void AnisotropicAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds, 
    ResponseSpan response, double era, std::size_t geg_order,
    std::size_t top_order, std::span<double> out)
{
    using ZernikeSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>;

    constexpr zest::RotationType rotation_type = zest::RotationType::COORDINATE;
    const auto& [boost_az, boost_colat, boost_speed]
        = coordinates::cartesian_to_spherical_phys(boost);
    const Vector<double, 3> euler_angles
        = euler_angles_to_align_z<rotation_type>(boost_az, boost_colat);

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), {geg_order}, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        ZernikeSpan::SubSpan rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
        out[i] = m_integrator_core.integrate(
                rotated_geg_zernike_grids, response[i], era, boost, 
                min_speeds[i], m_wigner_d_pi2);
}

IsotropicTransverseAngleIntegrator::IsotropicTransverseAngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 4),
    m_rotor(dist_order + 4),
    m_geg_zernike_exp(dist_order + 2),
    m_geg_zernike_exp_x(dist_order + 3),
    m_geg_zernike_exp_y(dist_order + 3),
    m_geg_zernike_exp_z(dist_order + 3),
    m_geg_zernike_exp_r2(dist_order + 4),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_rotated_trans_geg_zernike_exp(dist_order + 4),
    m_multiplier(dist_order + 4),
    m_integrator_core(dist_order + 4),
    m_dist_order(dist_order) {}

void IsotropicTransverseAngleIntegrator::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t const_geg_order = dist_order + 2;
    const std::size_t linear_geg_order = dist_order + 3;
    const std::size_t trans_geg_order = dist_order + 4;
    m_wigner_d_pi2.expand(trans_geg_order);
    m_rotor.expand(trans_geg_order);
    m_geg_zernike_exp.resize(const_geg_order);
    m_geg_zernike_exp_x.resize(linear_geg_order);
    m_geg_zernike_exp_y.resize(linear_geg_order);
    m_geg_zernike_exp_z.resize(linear_geg_order);
    m_geg_zernike_exp_r2.resize(trans_geg_order);
    m_rotated_geg_zernike_exp.resize(const_geg_order);
    m_rotated_trans_geg_zernike_exp.resize(trans_geg_order);
    m_multiplier.expand(dist_order);
    m_integrator_core.resize(trans_geg_order);
    m_dist_order = dist_order;
}

void IsotropicTransverseAngleIntegrator::integrate(
    DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<std::array<double, 2>, 2> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);
    for (std::size_t i = 0; i < boosts.size(); ++i)
        integrate(boosts[i], min_speeds, out[i]);
}

void IsotropicTransverseAngleIntegrator::integrate(
    DistributionSpan distribution, const Vector<double, 3>& boost, std::span<const double> min_speeds, std::span<std::array<double, 2>> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);
    integrate(boost, min_speeds, out);
}

void IsotropicTransverseAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds,
    std::span<std::array<double, 2>> out)
{
    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    std::ranges::copy(
        m_geg_zernike_exp_r2.flatten(),
        m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        dot(boost, boost), m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[2], m_geg_zernike_exp_z.flatten());

    constexpr zest::RotationType rotation_type = zest::RotationType::COORDINATE;
    const auto& [boost_az, boost_colat, boost_speed]
        = coordinates::cartesian_to_spherical_phys(boost);
    const Vector<double, 3> euler_angles
        = euler_angles_to_align_z<rotation_type>(boost_az, boost_colat);

    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);
    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, rotation_type);

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_rotated_geg_zernike_exp, m_rotated_trans_geg_zernike_exp, boost_speed, min_speeds[i]);
}

AnisotropicTransverseAngleIntegrator::AnisotropicTransverseAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 4, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_geg_zernike_exp_x(dist_order + 3),
    m_geg_zernike_exp_y(dist_order + 3),
    m_geg_zernike_exp_z(dist_order + 3),
    m_geg_zernike_exp_r2(dist_order + 4),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_rotated_trans_geg_zernike_exp(dist_order + 4),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(dist_order + 2, resp_order, trunc_order)),
    m_rotated_trans_geg_zernike_grids(
        geg_zernike_grids_size(dist_order + 4, resp_order, trunc_order)),
    m_multiplier(dist_order),
    m_integrator_core(
        dist_order + 4, resp_order,
        std::min(dist_order + 4 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

void AnisotropicTransverseAngleIntegrator::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 4, resp_order));
    
    if (dist_order != m_dist_order)
    {
        m_multiplier.expand(dist_order);
        m_geg_zernike_exp.resize(dist_order + 2);
        m_geg_zernike_exp_x.resize(dist_order + 3);
        m_geg_zernike_exp_y.resize(dist_order + 3);
        m_geg_zernike_exp_z.resize(dist_order + 3);
        m_geg_zernike_exp_r2.resize(dist_order + 4);
        m_rotated_geg_zernike_exp.resize(dist_order + 2);
        m_rotated_trans_geg_zernike_exp.resize(dist_order + 4);
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                        dist_order + 2, resp_order, trunc_order));
        m_rotated_trans_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                        dist_order + 4, resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 4, resp_order,
                std::min(dist_order + 4 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AnisotropicTransverseAngleIntegrator::integrate(
    DistributionSpan distribution, std::span<const Vector<double, 3>> boosts, 
    std::span<const double> min_speeds, ResponseSpan response,
    std::span<const double> era, zest::MDSpan<std::array<double, 2>, 2> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    for (std::size_t i = 0; i < boosts.size(); ++i)
        integrate(
                boosts[i], min_speeds, response, era[i], geg_order, top_order, 
                out[i]);
}

void AnisotropicTransverseAngleIntegrator::integrate(
    DistributionSpan distribution, const Vector<double, 3>& boost, 
    std::span<const double> min_speeds, ResponseSpan response, double era, 
    std::span<std::array<double, 2>> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    integrate(boost, min_speeds, response, era, geg_order, top_order, out);
}

void AnisotropicTransverseAngleIntegrator::integrate(
    const Vector<double, 3>& boost, std::span<const double> min_speeds, 
    ResponseSpan response, double era, std::size_t geg_order,
    std::size_t top_order, std::span<std::array<double, 2>> out)
{
    using ZernikeSpan = zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>;

    const double boost_sq = dot(boost, boost);

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), {geg_order}, top_order);

    constexpr zest::RotationType rotation_type = zest::RotationType::COORDINATE;
    const auto& [boost_az, boost_colat, boost_speed]
        = coordinates::cartesian_to_spherical_phys(boost);
    const Vector<double, 3> euler_angles
        = euler_angles_to_align_z<rotation_type>(boost_az, boost_colat);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        ZernikeSpan::SubSpan rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_trans_geg_zernike_grids(
            m_rotated_trans_geg_zernike_grids.data(), {geg_order}, top_order);

    std::ranges::copy(
            m_geg_zernike_exp_r2.flatten(),
            m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        boost_sq, m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*boost[2], m_geg_zernike_exp_z.flatten());
    
    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t n = 0; n < m_rotated_trans_geg_zernike_exp.order(); ++n)
        m_glq_transformer.backward_transform(
                m_rotated_trans_geg_zernike_exp[n], rotated_trans_geg_zernike_grids[n]);

    for (std::size_t i = 0; i < min_speeds.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                rotated_geg_zernike_grids, rotated_trans_geg_zernike_grids, 
                response[i], era, boost, min_speeds[i], m_wigner_d_pi2);
}

} // namespace zebra