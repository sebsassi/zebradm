/*
Copyright (c) 2024-2026 Sebastian Sassi

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

AngleIntegrator<DistType::iso, RespType::iso>::AngleIntegrator(std::size_t radon_order):
    m_integrator_core{radon_order},
    m_radon_order{radon_order} {}

void AngleIntegrator<DistType::iso, RespType::iso>::resize(std::size_t radon_order)
{
    if (radon_order == m_radon_order) return;
    m_integrator_core.resize(radon_order);
    m_radon_order = radon_order;
}

void AngleIntegrator<DistType::iso, RespType::iso>::integrate(
    IsotropicRadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    std::span<const la::Vector<double, 3>> offsets, std::span<const double> shells,
    zest::DynamicMDSpan<double, 2> out)
{
    resize(distribution_radon_transform.order());
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        const double offset_len = la::norm(offsets[i]);
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate(
                    distribution_radon_transform[0], offset_len, shells[j]);
    }
}

void AngleIntegrator<DistType::iso, RespType::iso>::integrate(
    IsotropicRadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    resize(distribution_radon_transform.order());
    const double offset_len = la::norm(offset);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                distribution_radon_transform[0], offset_len, shells[i]);
}

AngleIntegrator<DistType::iso, RespType::aniso>::AngleIntegrator(
    std::size_t radon_order, std::size_t resp_order):
    m_wigner_d_pi2{resp_order},
    m_integrator_core{radon_order, resp_order},
    m_radon_order{radon_order},
    m_resp_order{resp_order} {}

void AngleIntegrator<DistType::iso, RespType::aniso>::resize(
    std::size_t radon_order, std::size_t resp_order)
{
    if (radon_order == m_radon_order && resp_order == m_resp_order) return;

    m_wigner_d_pi2.expand(resp_order);
    m_integrator_core.resize(radon_order, resp_order);
    m_resp_order = resp_order;
    m_radon_order = radon_order;
}

void AngleIntegrator<DistType::iso, RespType::aniso>::integrate(
    IsotropicRadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    SHVectorSpan<const double> response,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::DynamicMDSpan<double, 2> out)
{
    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate(
                    distribution_radon_transform[0], response[j], offsets[i], rotation_angles[i], 
                    shells[j], m_wigner_d_pi2);
    }
}


void AngleIntegrator<DistType::iso, RespType::aniso>::integrate(
    IsotropicRadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<double> out)
{
    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                distribution_radon_transform[0], response[i], offset, rotation_angle, 
                shells[i], m_wigner_d_pi2);
}

AngleIntegrator<DistType::aniso, RespType::iso>::AngleIntegrator(
    std::size_t radon_order):
    m_wigner_d_pi2{radon_order},
    m_rotor{radon_order},
    m_rotated_radon_transform_exp{radon_order},
    m_integrator_core{radon_order},
    m_radon_order{radon_order} {}

void AngleIntegrator<DistType::aniso, RespType::iso>::resize(std::size_t radon_order)
{
    if (radon_order == m_radon_order) return;
    m_wigner_d_pi2.expand(radon_order);
    m_rotor.expand(radon_order);
    m_rotated_radon_transform_exp.reshape(radon_order);
    m_integrator_core.resize(radon_order);
    m_radon_order = radon_order;
}

void AngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    RadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> shells, zest::DynamicMDSpan<double, 2> out)
{
    assert(
        offsets.size() == out.extent(0)
        && shells.size() == out.extent(1));

    resize(distribution_radon_transform.order());
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(distribution_radon_transform, offsets[i], shells, out[i]);
}

void AngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    RadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    assert(shells.size() == out.size());
    resize(distribution_radon_transform.order());

    std::ranges::copy(
        distribution_radon_transform.flatten(),
        m_rotated_radon_transform_exp.flatten().begin());

    constexpr zest::RotationType rotation_type = zest::RotationType::passive;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    m_rotor.rotate<rotation_type>(
            m_rotated_radon_transform_exp, m_wigner_d_pi2, euler_angles);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_rotated_radon_transform_exp, offset_len, shells[i]);
}

namespace
{

[[nodiscard]] constexpr std::size_t
radon_zernike_grids_size(
    std::size_t num_grids, std::size_t grid_order, std::size_t trunc_order) noexcept
{
    return zest::st::SphereGLQGridVectorSpan<double>::size(num_grids, std::min(grid_order, trunc_order));
}

[[nodiscard]] constexpr std::size_t
zernike_expansion_sh_span_size(std::size_t order)
{
    return zest::subspan<ZernikeExpansion<double>, 1>::size(order);
}

} // namespace

AngleIntegrator<DistType::aniso, RespType::aniso>::AngleIntegrator(
    std::size_t radon_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2{std::max(radon_order, resp_order)},
    m_rotated_radon_transform_exp(zernike_expansion_sh_span_size(radon_order)),
    m_rotated_radon_transform_grids(
        radon_zernike_grids_size(radon_order, radon_order + resp_order, trunc_order)),
    m_integrator_core{
        radon_order, resp_order,
        std::min(radon_order + resp_order, trunc_order)},
    m_radon_order{radon_order},
    m_resp_order{resp_order},
    m_trunc_order{trunc_order} {}

void AngleIntegrator<DistType::aniso, RespType::aniso>::resize(
    std::size_t radon_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (std::max(m_radon_order, m_resp_order) < std::max(radon_order, resp_order))
        m_wigner_d_pi2.expand(std::max(radon_order, resp_order));

    if (radon_order != m_radon_order)
        m_rotated_radon_transform_exp.resize(zernike_expansion_sh_span_size(radon_order));

    if (radon_order != m_radon_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_radon_transform_grids.resize(
                radon_zernike_grids_size(radon_order, radon_order + resp_order, trunc_order));
        m_integrator_core.resize(
                radon_order, resp_order, std::min(radon_order + resp_order, trunc_order));
    }

    m_radon_order = radon_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    SHVectorSpan<const double> response,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::DynamicMDSpan<double, 2> out, std::size_t trunc_order)
{
    assert(
        offsets.size() == out.extent(0)
        && rotation_angles.size() == out.extent(0)
        && shells.size() == out.extent(1));

    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order, trunc_order);
    const std::size_t top_order = std::min(radon_order + resp_order, trunc_order);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(
                distribution_radon_transform, response, offsets[i], rotation_angles[i], shells, radon_order,
                top_order, out[i]);
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<double> out,
    std::size_t trunc_order)
{
    assert(shells.size() == out.size());

    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order, trunc_order);
    const std::size_t top_order = std::min(radon_order + resp_order, trunc_order);

    integrate(distribution_radon_transform, response, offset, rotation_angle, shells, radon_order, top_order, out);
}

void AngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::identity> distribution_radon_transform,
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::size_t radon_order,
    std::size_t top_order, std::span<double> out)
{
    assert(shells.size() == out.size());

    constexpr zest::RotationType rotation_type = zest::RotationType::passive;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_radon_zernike_grids(
            m_rotated_radon_transform_grids.data(), radon_order, top_order);

    for (std::size_t n = 0; n < radon_order; ++n)
    {
        std::ranges::copy(
                distribution_radon_transform[0, n].flatten(),
                m_rotated_radon_transform_exp.begin());
        zest::subspan<ZernikeSpan<double>, 1>
        rotated_radon_zernike_exp{m_rotated_radon_transform_exp.data(), n + 1};

        m_rotor.rotate<rotation_type>(
                rotated_radon_zernike_exp, m_wigner_d_pi2, euler_angles);
        m_glq_transformer.backward_transform(
                rotated_radon_zernike_exp, rotated_radon_zernike_grids[n]);
    }

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                rotated_radon_zernike_grids, response[i], offset, rotation_angle, 
                shells[i], m_wigner_d_pi2);
}

TransverseAngleIntegrator<DistType::iso, RespType::iso>::TransverseAngleIntegrator(std::size_t radon_order):
    m_integrator_core{radon_order},
    m_radon_order{radon_order} {}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::resize(std::size_t radon_order)
{
    if (radon_order == m_radon_order) return;
    m_integrator_core.resize(radon_order);
    m_radon_order = radon_order;
}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicRadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> shells, zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    resize(distribution_radon_transform.order());
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        const double offset_len = la::norm(offsets[i]);
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate_transverse(
                    distribution_radon_transform, offset_len, shells[j]);
    }

}

void TransverseAngleIntegrator<DistType::iso, RespType::iso>::integrate(
        IsotropicRadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
        const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    resize(distribution_radon_transform.order());
    const double offset_len = la::norm(offset);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                distribution_radon_transform, offset_len, shells[i]);

}

TransverseAngleIntegrator<DistType::iso, RespType::aniso>::TransverseAngleIntegrator(
    std::size_t radon_order, std::size_t resp_order):
    m_wigner_d_pi2{resp_order},
    m_integrator_core{radon_order, resp_order},
    m_radon_order{radon_order},
    m_resp_order{resp_order} {}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::resize(
    std::size_t radon_order, std::size_t resp_order)
{
    if (resp_order != m_resp_order) return;
    m_wigner_d_pi2.expand(resp_order);
    m_integrator_core.resize(radon_order, resp_order);
    m_radon_order = radon_order;
    m_resp_order = resp_order;
}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicRadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
        SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            out[i, j] = m_integrator_core.integrate_transverse(
                    distribution_radon_transform, response[j], offsets[i],
                    rotation_angles[i], shells[j], m_wigner_d_pi2);
    }

}

void TransverseAngleIntegrator<DistType::iso, RespType::aniso>::integrate(
        IsotropicRadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
        SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order);
    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                distribution_radon_transform, response[i], offset, rotation_angle,
                shells[i], m_wigner_d_pi2);

}

TransverseAngleIntegrator<DistType::aniso, RespType::iso>::TransverseAngleIntegrator(
    std::size_t radon_order):
    m_wigner_d_pi2{radon_order + 2},
    m_rotor{radon_order + 2},
    m_rotated_radon_transform_exp{radon_order},
    m_rotated_trans_radon_transform_exp{radon_order + 2},
    m_integrator_core{radon_order + 2},
    m_radon_order{radon_order} {}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::resize(std::size_t radon_order)
{
    if (radon_order == m_radon_order) return;
    m_wigner_d_pi2.expand(radon_order + 2);
    m_rotor.expand(radon_order + 2);
    m_rotated_radon_transform_exp.reshape(radon_order);
    m_rotated_trans_radon_transform_exp.reshape(radon_order + 2);
    m_integrator_core.resize(radon_order + 2);
    m_radon_order = radon_order;
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    ZernikeSpan<const double> distribution,
    std::span<const la::Vector<double, 3>> offsets, std::span<const double> shells,
    zest::DynamicMDSpan<std::array<double, 2>, 2> out)
{
    assert(
        offsets.size() == out.extent(0)
        && shells.size() == out.extent(1));

    resize(distribution_radon_transform.order());
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(distribution_radon_transform, offsets[i], shells, out[i]);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    ZernikeSpan<const double> distribution,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

    resize(distribution_radon_transform.order());
    integrate(distribution_radon_transform, offset, shells, out);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::iso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    const la::Vector<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

    std::ranges::copy(
        distribution_radon_transform[0].flatten(),
        m_rotated_radon_transform_exp.flatten().begin());

    evaluate_transverse_radon_transform(
        distribution_radon_transform, offset, m_rotated_trans_radon_transform_exp);

    constexpr zest::RotationType rotation_type = zest::RotationType::passive;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    m_rotor.rotate<rotation_type>(
            m_rotated_radon_transform_exp, m_wigner_d_pi2, euler_angles);
    m_rotor.rotate<rotation_type>(
            m_rotated_trans_radon_transform_exp, m_wigner_d_pi2, euler_angles);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_rotated_radon_transform_exp, m_rotated_trans_radon_transform_exp, offset_len, shells[i]);
}

TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::TransverseAngleIntegrator(
    std::size_t radon_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2{std::max(radon_order + 2, resp_order)},
    m_rotated_radon_transform_exp{zernike_expansion_sh_span_size(radon_order)},
    m_rotated_trans_radon_transform_exp{radon_order + 2},
    m_rotated_radon_transform_grids{
        radon_zernike_grids_size(
            radon_order, radon_order + 2 + resp_order, trunc_order)},
    m_rotated_trans_radon_transform_grids{
        radon_zernike_grids_size(
            radon_order + 2, radon_order + 2 + resp_order, trunc_order)},
    m_integrator_core{
        radon_order + 2, resp_order,
        std::min(radon_order + 2 + resp_order, trunc_order)},
    m_radon_order{radon_order},
    m_resp_order{resp_order},
    m_trunc_order{trunc_order} {}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::resize(
    std::size_t radon_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (std::max(m_radon_order, m_resp_order) < std::max(radon_order, resp_order))
        m_wigner_d_pi2.expand(std::max(radon_order + 2, resp_order));

    if (radon_order != m_radon_order)
    {
        m_rotated_radon_transform_exp.resize(zernike_expansion_sh_span_size(radon_order));
        m_rotated_trans_radon_transform_exp.reshape(radon_order + 2);
    }

    if (radon_order != m_radon_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_radon_transform_grids.resize(
                radon_zernike_grids_size(radon_order, radon_order + 2 + resp_order, trunc_order));
        m_rotated_trans_radon_transform_grids.resize(
                radon_zernike_grids_size(
                    radon_order + 2, radon_order + 2 + resp_order, trunc_order));
        m_integrator_core.resize(
                radon_order + 2, resp_order, std::min(radon_order + 2 + resp_order, trunc_order));
    }

    m_radon_order = radon_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    SHVectorSpan<const double> response,
    std::span<const la::Vector<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::DynamicMDSpan<std::array<double, 2>, 2> out, std::size_t trunc_order)
{
    assert(
        offsets.size() == out.extent(0)
        && rotation_angles.size() == out.extent(0)
        && shells.size() == out.extent(1));

    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order, trunc_order);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(distribution_radon_transform, response, offsets[i], rotation_angles[i], shells, out[i]);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out,
    std::size_t trunc_order)
{
    const std::size_t radon_order = distribution_radon_transform.order();
    const std::size_t resp_order = std::get<0>(std::get<1>(response.extents()));
    resize(radon_order, resp_order, trunc_order);

    integrate(distribution_radon_transform, response, offset, rotation_angle, shells, out);
}

void TransverseAngleIntegrator<DistType::aniso, RespType::aniso>::integrate(
    RadonMomentSpan<const double, MomentCategory::transverse> distribution_radon_transform,
    SHVectorSpan<const double> response,
    const la::Vector<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    assert(shells.size() == out.size());

    constexpr zest::RotationType rotation_type = zest::RotationType::passive;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    const std::size_t trans_radon_order = m_radon_order + 2;
    const std::size_t top_order = std::min(trans_radon_order + m_resp_order, m_trunc_order);

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_radon_zernike_grids(
            m_rotated_radon_transform_grids.data(), m_radon_order, top_order);

    for (std::size_t n = 0; n < m_radon_order; ++n)
    {
        std::ranges::copy(
                distribution_radon_transform[0, n].flatten(),
                m_rotated_radon_transform_exp.begin());
        ZernikeSpan<double>::subspan_type<1>
        rotated_radon_zernike_exp{m_rotated_radon_transform_exp.data(), n + 1};

        m_rotor.rotate<rotation_type>(
                rotated_radon_zernike_exp, m_wigner_d_pi2, euler_angles);
        m_glq_transformer.backward_transform(
                rotated_radon_zernike_exp, rotated_radon_zernike_grids[n]);
    }

    zest::st::SphereGLQGridVectorSpan<double>
    rotated_trans_radon_zernike_grids(
            m_rotated_trans_radon_transform_grids.data(), trans_radon_order, top_order);

    evaluate_transverse_radon_transform(
        distribution_radon_transform, offset, m_rotated_trans_radon_transform_exp);

    m_rotor.rotate<rotation_type>(
            m_rotated_trans_radon_transform_exp, m_wigner_d_pi2, euler_angles);

    for (std::size_t n = 0; n < m_rotated_trans_radon_transform_exp.order(); ++n)
        m_glq_transformer.backward_transform(
                m_rotated_trans_radon_transform_exp[n], rotated_trans_radon_zernike_grids[n]);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                rotated_radon_zernike_grids, rotated_trans_radon_zernike_grids,
                response[i], offset, rotation_angle, shells[i], m_wigner_d_pi2);
}

} // namespace zdm::zebra
