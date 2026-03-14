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
#include "zebra_angle_integrator.hpp"

#include <zest/md_span.hpp>
#include <zest/rotor.hpp>
#include <zest/sh_glq_transformer.hpp>
#include <zest/zernike_conventions.hpp>
#include <zest/zernike_expansion.hpp>

#include "coordinate_transforms.hpp"

#include "radon_util.hpp"
#include "types.hpp"
#include "utility.hpp"
#include "zebra_radon.hpp"

namespace zdm::zebra
{

AngleIntegrator<DistType::iso, RespType::iso>::AngleIntegrator(std::size_t dist_order):
    m_geg_zernike_exp(dist_order + 2),
    m_integrator_core(dist_order + 2),
    m_dist_order(dist_order) {}

void AngleIntegrator<DistType::iso, RespType::iso>::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_geg_zernike_exp.reshape(geg_order);
    m_integrator_core.resize(geg_order);
    m_dist_order = dist_order;
}

void AngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicZernikeSpan<const double> distribution,
        std::span<const la::Vector<double, 3>> offsets, std::span<const double> shells,
        zest::DynamicMDSpan<double, 2> out)
{
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        const double offset_len = la::length(offsets[i]);
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate(
                    m_geg_zernike_exp, offset_len, shells[j]);
    }
}

void AngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<double> out)
{
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    const double offset_len = la::length(offset);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_geg_zernike_exp, offset_len, shells[i]);
}

AngleIntegrator<DistType::iso, RespType::aniso>::AngleIntegrator(
    std::size_t dist_order, std::size_t resp_order):
    m_wigner_d_pi2(resp_order),
    m_geg_zernike_exp(dist_order + 2),
    m_integrator_core(dist_order + 2, resp_order),
    m_dist_order(dist_order), m_resp_order(resp_order) {}

void AngleIntegrator<DistType::iso, RespType::aniso>::resize(
    std::size_t dist_order, std::size_t resp_order)
{
    if (dist_order == m_dist_order && resp_order == m_resp_order) return;

    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(resp_order);
    m_integrator_core.resize(geg_order, resp_order);
    m_resp_order = resp_order;
    if (dist_order == m_dist_order) return;

    m_geg_zernike_exp.reshape(geg_order);
    m_dist_order = dist_order;
}


void AngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<double, 2> out)
{
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate(
                    m_geg_zernike_exp, response[j], offsets[i], rotation_angles[i], 
                    shells[j], m_wigner_d_pi2);
    }
}


void AngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<double> out)
{
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_geg_zernike_exp, response[i], offset, rotation_angle, 
                shells[i], m_wigner_d_pi2);
}

AngleIntegrator<DistType::aniso, RespType::iso>::AngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 2),
    m_rotor(dist_order + 2),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_integrator_core(dist_order + 2),
    m_dist_order(dist_order) {}

void AngleIntegrator<DistType::aniso, RespType::iso>::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(geg_order);
    m_rotor.expand(geg_order);
    m_geg_zernike_exp.reshape(geg_order);
    m_rotated_geg_zernike_exp.reshape(geg_order);
    m_integrator_core.resize(geg_order);
    m_dist_order = dist_order;
}

void AngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    ZernikeSpan<const double> distribution,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> shells, zest::DynamicMDSpan<double, 2> out)
{
    assert(
        offsets.size() == out.extent(0)
        && shells.size() == out.extent(1));

    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(offsets[i], shells, out[i]);
}

void AngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    ZernikeSpan<const double> distribution,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    assert(shells.size() == out.size());
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    integrate(offset, shells, out);
}

void AngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    assert(shells.size() == out.size());

    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_rotated_geg_zernike_exp, offset_len, shells[i]);
}

namespace
{

[[nodiscard]] constexpr std::size_t geg_zernike_grids_size(
    std::size_t num_grids, std::size_t grid_order, std::size_t trunc_order
) noexcept
{
    return num_grids*zest::st::SphereGLQGridSpan<double>::size(std::min(grid_order, trunc_order));
}

[[nodiscard]] constexpr std::size_t zernike_expansion_sh_span_size(
    std::size_t order)
{
    return ZernikeExpansion<double>::subspan_type<1>::size(order);
}

} // namespace

AngleIntegrator<DistType::aniso, RespType::aniso>::AngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 2, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(zernike_expansion_sh_span_size(dist_order + 2)),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 2, dist_order + 2 + resp_order, trunc_order)),
    m_integrator_core(
        dist_order + 2, resp_order,
        std::min(dist_order + 2 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

void AngleIntegrator<DistType::aniso, RespType::aniso>::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 2, resp_order));

    if (dist_order != m_dist_order)
    {
        m_geg_zernike_exp.reshape(dist_order + 2);
        m_rotated_geg_zernike_exp.resize(
                zernike_expansion_sh_span_size(dist_order + 2));
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 2, dist_order + 2 + resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 2, resp_order,
                std::min(dist_order + 2 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::DynamicMDSpan<double, 2> out, std::size_t trunc_order)
{
    assert(
        offsets.size() == out.extent(0)
        && rotation_angles.size() == out.extent(0)
        && shells.size() == out.extent(1));

    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(
                response, offsets[i], rotation_angles[i], shells, geg_order,
                top_order, out[i]);
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<double> out,
    std::size_t trunc_order)
{
    assert(shells.size() == out.size());

    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    integrate(response, offset, rotation_angle, shells, geg_order, top_order, out);
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::size_t geg_order,
    std::size_t top_order, std::span<double> out)
{
    assert(shells.size() == out.size());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), geg_order, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        ZernikeSpan<double>::subspan_type<1> rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);

        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                rotated_geg_zernike_grids, response[i], offset, rotation_angle, 
                shells[i], m_wigner_d_pi2);
}

TransverseAngleIntegrator<DistType::iso, RespType::iso>::TransverseAngleIntegrator(std::size_t dist_order):
    m_transverse_geg_zernike_exp_components(dist_order + 4),
    m_transverse_radon_helper(dist_order),
    m_integrator_core(dist_order + 2),
    m_dist_order(dist_order) {}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_transverse_geg_zernike_exp_components.reshape(geg_order);
    m_integrator_core.resize(geg_order);
    m_dist_order = dist_order;
}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicZernikeSpan<const double> distribution, std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> shells, zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    m_transverse_radon_helper
        .evaluate_transverse_components(distribution, m_transverse_geg_zernike_exp_components);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        const double offset_len = la::length(offsets[i]);
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate_transverse(
                    m_transverse_geg_zernike_exp_components, offset_len, shells[j]);
    }

}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    m_transverse_radon_helper
        .evaluate_transverse_components(distribution, m_transverse_geg_zernike_exp_components);
    const double offset_len = la::length(offset);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_transverse_geg_zernike_exp_components, offset_len, shells[i]);

}

TransverseAngleIntegrator<DistType::iso, RespType::aniso>::TransverseAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order):
    m_wigner_d_pi2(resp_order),
    m_transverse_geg_zernike_exp_components(dist_order + 4),
    m_transverse_radon_helper(dist_order),
    m_integrator_core(dist_order + 2, resp_order),
    m_dist_order(dist_order), m_resp_order(resp_order) {}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::resize(
    std::size_t dist_order, std::size_t resp_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(resp_order);
    m_transverse_geg_zernike_exp_components.reshape(geg_order);
    m_integrator_core.resize(geg_order, resp_order);
    m_dist_order = dist_order;
    m_resp_order = resp_order;
}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    m_transverse_radon_helper
        .evaluate_transverse_components(distribution, m_transverse_geg_zernike_exp_components);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate_transverse(
                    m_transverse_geg_zernike_exp_components, response[j], offsets[i],
                    rotation_angles[i], shells[j], m_wigner_d_pi2);
    }

}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    m_transverse_radon_helper
        .evaluate_transverse_components(distribution, m_transverse_geg_zernike_exp_components);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_transverse_geg_zernike_exp_components, response[i], offset, rotation_angle,
                shells[i], m_wigner_d_pi2);

}

TransverseAngleIntegrator<DistType::aniso, RespType::iso>::TransverseAngleIntegrator(
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

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t const_geg_order = dist_order + 2;
    const std::size_t linear_geg_order = dist_order + 3;
    const std::size_t trans_geg_order = dist_order + 4;
    m_wigner_d_pi2.expand(trans_geg_order);
    m_rotor.expand(trans_geg_order);
    m_geg_zernike_exp.reshape(const_geg_order);
    m_geg_zernike_exp_x.reshape(linear_geg_order);
    m_geg_zernike_exp_y.reshape(linear_geg_order);
    m_geg_zernike_exp_z.reshape(linear_geg_order);
    m_geg_zernike_exp_r2.reshape(trans_geg_order);
    m_rotated_geg_zernike_exp.reshape(const_geg_order);
    m_rotated_trans_geg_zernike_exp.reshape(trans_geg_order);
    m_multiplier.expand(dist_order);
    m_integrator_core.resize(trans_geg_order);
    m_dist_order = dist_order;
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    ZernikeSpan<const double> distribution,
    std::span<const la::Vector<double, 3>> offsets, std::span<const double> shells,
    zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    assert(
        offsets.size() == out.extent(0)
        && shells.size() == out.extent(1));

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
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(offsets[i], shells, out[i]);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    ZernikeSpan<const double> distribution,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

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
    integrate(offset, shells, out);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    std::ranges::copy(
        m_geg_zernike_exp_r2.flatten(),
        m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        la::dot(offset, offset), m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[2], m_geg_zernike_exp_z.flatten());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);
    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, rotation_type);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_rotated_geg_zernike_exp, m_rotated_trans_geg_zernike_exp, offset_len, shells[i]);
}

TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::TransverseAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 4, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_geg_zernike_exp_x(dist_order + 3),
    m_geg_zernike_exp_y(dist_order + 3),
    m_geg_zernike_exp_z(dist_order + 3),
    m_geg_zernike_exp_r2(dist_order + 4),
    m_rotated_geg_zernike_exp(zernike_expansion_sh_span_size(dist_order + 2)),
    m_rotated_trans_geg_zernike_exp(dist_order + 4),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 2, dist_order + 4 + resp_order, trunc_order)),
    m_rotated_trans_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 4, dist_order + 4 + resp_order, trunc_order)),
    m_multiplier(dist_order + 4),
    m_integrator_core(
        dist_order + 4, resp_order,
        std::min(dist_order + 4 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 4, resp_order));

    if (dist_order != m_dist_order)
    {
        m_multiplier.expand(dist_order + 4);
        m_geg_zernike_exp.reshape(dist_order + 2);
        m_geg_zernike_exp_x.reshape(dist_order + 3);
        m_geg_zernike_exp_y.reshape(dist_order + 3);
        m_geg_zernike_exp_z.reshape(dist_order + 3);
        m_geg_zernike_exp_r2.reshape(dist_order + 4);
        m_rotated_geg_zernike_exp.resize(
                zernike_expansion_sh_span_size(dist_order + 2));
        m_rotated_trans_geg_zernike_exp.reshape(dist_order + 4);
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 2, dist_order + 4 + resp_order, trunc_order));
        m_rotated_trans_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 4, dist_order + 4 + resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 4, resp_order,
                std::min(dist_order + 4 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::DynamicMDSpan<std::array<double, 2>, 2> out, std::size_t trunc_order)
{
    assert(
        offsets.size() == out.extent(0)
        && rotation_angles.size() == out.extent(0)
        && shells.size() == out.extent(1));

    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(response, offsets[i], rotation_angles[i], shells, out[i]);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    integrate(response, offset, rotation_angle, shells, out);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    const std::size_t geg_order = m_dist_order + 2;
    const std::size_t trans_geg_order = m_dist_order + 4;
    const std::size_t top_order = std::min(trans_geg_order + m_resp_order, m_trunc_order);

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), geg_order, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        ZernikeSpan<double>::subspan_type<1> rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_trans_geg_zernike_grids(
            m_rotated_trans_geg_zernike_grids.data(), trans_geg_order, top_order);

    std::ranges::copy(
            m_geg_zernike_exp_r2.flatten(),
            m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        offset_len*offset_len, m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[2], m_geg_zernike_exp_z.flatten());

    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t n = 0; n < m_rotated_trans_geg_zernike_exp.order(); ++n)
        m_glq_transformer.backward_transform(
                m_rotated_trans_geg_zernike_exp[n], rotated_trans_geg_zernike_grids[n]);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                rotated_geg_zernike_grids, rotated_trans_geg_zernike_grids, 
                response[i], offset, rotation_angle, shells[i], m_wigner_d_pi2);
}

} // namespace zdm::zebra
