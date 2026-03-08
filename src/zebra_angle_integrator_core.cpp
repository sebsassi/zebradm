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
#include "zebra_angle_integrator_core.hpp"

#include <zest/grid_evaluator.hpp>

#include "coordinate_transforms.hpp"
#include "radon_util.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace zdm::zebra::detail
{

AngleIntegratorCore<DistType::iso, RespType::iso>::AngleIntegratorCore(std::size_t geg_order):
    m_legendre_integral_recursion(geg_order + 2),
    m_legendre_integrals(geg_order + 2) {}

void AngleIntegratorCore<DistType::iso, RespType::iso>::resize(std::size_t geg_order)
{
    m_legendre_integral_recursion.expand(geg_order);
    m_legendre_integrals.resize(geg_order);
}

double AngleIntegratorCore<DistType::iso, RespType::iso>::integrate(
    IsotropicZernikeSpan<const double> geg_zernike_exp, double offset_len, double shell)
{
    if (std::fabs(shell) > 1.0 + offset_len) return 0.0;

    const double xmin = std::max(-1.0, shell - offset_len);
    const double xmax = std::min(1.0, shell + offset_len);
    const la::Vector<double, 2> x = {xmin, xmax};
    m_legendre_integral_recursion.generate(std::span(m_legendre_integrals), x);

    la::Vector<double, 2> res = geg_zernike_exp[0]*m_legendre_integrals[0];
    for (auto n : geg_zernike_exp.indices(2))
        res += geg_zernike_exp[n]*m_legendre_integrals[n];

    constexpr double two_pi_sq = (2.0*std::numbers::pi)*(2.0*std::numbers::pi);
    return two_pi_sq*(res[1] - res[0])/offset_len;
}

std::array<double, 2> AngleIntegratorCore<DistType::iso, RespType::iso>::integrate_transverse(
    IsotropicZernikeSpan<const double, 3> trans_geg_zernike_exp, double offset_len, double shell)
{
    if (std::fabs(shell) > 1.0 + offset_len) return {};

    const double xmin = std::max(-1.0, shell - offset_len);
    const double xmax = std::min(1.0, shell + offset_len);
    const la::Vector<double, 2> x = {xmin, xmax};
    m_legendre_integral_recursion.generate(std::span(m_legendre_integrals), x);

    std::array<la::Vector<double, 2>, 3> res = {
        trans_geg_zernike_exp[0, 0]*m_legendre_integrals[0],
        trans_geg_zernike_exp[0, 1]*m_legendre_integrals[1],
        trans_geg_zernike_exp[0, 2]*m_legendre_integrals[0]
    };
    for (std::size_t n = 2; n < trans_geg_zernike_exp.order() - 2; n += 2)
    {
        res[0] += trans_geg_zernike_exp[n, 0]*m_legendre_integrals[n];
        res[1] += trans_geg_zernike_exp[n, 1]*m_legendre_integrals[n + 1];
        res[2] += trans_geg_zernike_exp[n, 2]*m_legendre_integrals[n];
    }

    const std::size_t nmax = util::even_floor(trans_geg_zernike_exp.order() - 1);
    res[0] += trans_geg_zernike_exp[nmax, 0]*m_legendre_integrals[nmax];

    la::Vector<double, 2> nontrans_res = res[2];
    la::Vector<double, 2> trans_res = res[0] - shell*res[1] + (offset_len*offset_len - shell*shell)*res[2];
    constexpr double two_pi_sq = (2.0*std::numbers::pi)*(2.0*std::numbers::pi);
    return {
        two_pi_sq*(nontrans_res[1] - nontrans_res[0])/offset_len,
        two_pi_sq*(trans_res[1] - trans_res[0])/offset_len
    };
}

AngleIntegratorCore<DistType::iso, RespType::aniso>::AngleIntegratorCore(
    std::size_t geg_order, std::size_t resp_order):
    m_rotor(resp_order),
    m_rotated_response_exp(resp_order),
    m_zonal_rotated_response_exp(resp_order),
    m_aff_leg_integrals(geg_order, resp_order),
    m_aff_leg_ylm_integrals(geg_order, resp_order),
    m_ylm_integral_norms(resp_order)
{
    for (std::size_t l = 0; l  < resp_order; ++l)
        m_ylm_integral_norms[l] = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

void AngleIntegratorCore<DistType::iso, RespType::aniso>::resize(
    std::size_t geg_order, std::size_t resp_order)
{
    m_rotor.expand(resp_order);
    m_rotated_response_exp.reshape(resp_order);
    m_zonal_rotated_response_exp.resize(resp_order);

    m_aff_leg_integrals.resize(geg_order, resp_order);
    m_aff_leg_ylm_integrals.reshape(geg_order, resp_order);
    m_ylm_integral_norms.resize(resp_order);
    for (std::size_t l = 0; l  < resp_order; ++l)
        m_ylm_integral_norms[l] = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

[[nodiscard]] double
AngleIntegratorCore<DistType::iso, RespType::aniso>::integrate(
    IsotropicZernikeSpan<const double> geg_zernike_exp,
    SHSpan<const double> response_exp,
    const la::Vector<double, 3>& offset, double rotation_angle, double shell,
    const zest::WignerdPiHalfCollection& wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);

    if (std::fabs(shell) > 1.0 + offset_len) return 0.0;

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);
    for (std::size_t l : m_rotated_response_exp.indices())
        m_zonal_rotated_response_exp[l] = m_rotated_response_exp[l, 0, 0];

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    double res = 0.0;
    for (std::size_t n : geg_zernike_exp.indices())
        res += geg_zernike_exp[n]*util::inner_product(
                std::span<double>(m_zonal_rotated_response_exp),
                m_aff_leg_ylm_integrals[n].flatten());

    return (2.0*std::numbers::pi)*res;
}

[[nodiscard]] std::array<double, 2>
AngleIntegratorCore<DistType::iso, RespType::aniso>::integrate_transverse(
    IsotropicZernikeSpan<const double, 3> trans_geg_zernike_exp, SHSpan<const double> response_exp,
    const la::Vector<double, 3>& offset, double rotation_angle, double shell,
    const zest::WignerdPiHalfCollection& wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    if (std::fabs(shell) > 1.0 + offset_len) return {0.0, 0.0};

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);
    for (std::size_t l : m_rotated_response_exp.indices())
        m_zonal_rotated_response_exp[l] = m_rotated_response_exp[l, 0, 0];

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    double trans = 0.0;
    double nontrans = 0.0;
    for (std::size_t n : trans_geg_zernike_exp.indices())
    {
        const std::array<double, 2> angle_integrals = {
            util::inner_product(
                std::span<double>(m_zonal_rotated_response_exp),
                m_aff_leg_ylm_integrals[n].flatten()),
            util::inner_product(
                std::span<double>(m_zonal_rotated_response_exp),
                m_aff_leg_ylm_integrals[n + 1].flatten())
        };

        trans += trans_geg_zernike_exp[n, 0]*angle_integrals[0]
            - shell*trans_geg_zernike_exp[n, 1]*angle_integrals[1];
        nontrans += trans_geg_zernike_exp[n, 2]*angle_integrals[2];
    }

    return {
        2.0*std::numbers::pi*nontrans,
        2.0*std::numbers::pi*(trans + (offset_len*offset_len - shell*shell)*nontrans)
    };
}

void AngleIntegratorCore<DistType::iso, RespType::aniso>::evaluate_aff_leg_ylm_integrals(
    double shell, double offset_len)
{
    m_aff_leg_integrals.integrals(m_aff_leg_ylm_integrals, shell, offset_len);

    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        util::mul(aff_leg_ylm_integrals_n.flatten(), std::span(m_ylm_integral_norms));
    }
}


AngleIntegratorCore<DistType::aniso, RespType::iso>::AngleIntegratorCore(
    std::size_t geg_order):
    m_aff_leg_integrals(geg_order, 0),
    m_aff_leg_ylm_integrals(geg_order, 0),
    m_ylm_integral_norms(geg_order)
{
    for (std::size_t l = 0; l  < geg_order; ++l)
        m_ylm_integral_norms[l]
            = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

void AngleIntegratorCore<DistType::aniso, RespType::iso>::resize(std::size_t geg_order)
{
    if (order() == geg_order) return;
    m_aff_leg_integrals.resize(geg_order, 0);
    m_aff_leg_ylm_integrals.reshape(geg_order, 0);
    m_ylm_integral_norms.resize(geg_order);
    for (std::size_t l = 0; l  < geg_order; ++l)
        m_ylm_integral_norms[l] = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

[[nodiscard]] double
AngleIntegratorCore<DistType::aniso, RespType::iso>::integrate(
    ZernikeSpan<const double> rotated_geg_zernike_exp,
    double offset_len, double shell)
{
    if (std::fabs(shell) > 1.0 + offset_len) return 0.0;

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    double res = 0;
    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto rotated_geg_n = rotated_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        for (std::size_t l = n & 1; l <= n; l += 2)
            res += rotated_geg_n[l, 0, 0]*aff_leg_ylm_integrals_n[l];
    }

    return (2.0*std::numbers::pi)*res;
}

[[nodiscard]] std::array<double, 2>
AngleIntegratorCore<DistType::aniso, RespType::iso>::integrate_transverse(
    ZernikeSpan<const double> rotated_geg_zernike_exp,
    ZernikeSpan<const double> rotated_trans_geg_zernike_exp,
    double offset_len, double shell)
{
    if (std::fabs(shell) > 1.0 + offset_len) return {0.0, 0.0};

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    // transverse contribution
    double trans = 0.0;
    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto rotated_trans_geg_n = rotated_trans_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        for (std::size_t l = n & 1; l <= n; l += 2)
            trans += rotated_trans_geg_n[l, 0, 0]*aff_leg_ylm_integrals_n[l];
    }

    TrapezoidSpan<const double>
    non_trans_aff_leg_ylm_integrals(
            m_aff_leg_ylm_integrals.data(),
            m_aff_leg_ylm_integrals.order() - 2,
            m_aff_leg_ylm_integrals.shape().extra_extent());

    // nontransverse contribution
    double nontrans = 0.0;
    for (std::size_t n : non_trans_aff_leg_ylm_integrals.indices())
    {
        auto rotated_geg_n = rotated_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = non_trans_aff_leg_ylm_integrals[n];

        for (std::size_t l = n & 1; l <= n; l += 2)
            nontrans += rotated_geg_n[l, 0, 0]*aff_leg_ylm_integrals_n[l];
    }

    const double shell_sq = shell*shell;
    return {(2.0*std::numbers::pi)*nontrans, (2.0*std::numbers::pi)*(trans - shell_sq*nontrans)};
}

void AngleIntegratorCore<DistType::aniso, RespType::iso>::evaluate_aff_leg_ylm_integrals(
    double shell, double offset_len)
{
    m_aff_leg_integrals.integrals(m_aff_leg_ylm_integrals, shell, offset_len);
    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        util::mul(aff_leg_ylm_integrals_n.flatten(), std::span(m_ylm_integral_norms));
    }
}

AngleIntegratorCore<DistType::aniso, RespType::aniso>::AngleIntegratorCore(
    std::size_t geg_order, std::size_t resp_order, std::size_t top_order):
    m_rotor(std::max(geg_order, resp_order)), m_glq_transformer(top_order),
    m_rotated_response_exp(resp_order), m_rotated_response_grid(top_order),
    m_aff_leg_integrals(geg_order, resp_order),
    m_aff_leg_ylm_integrals(geg_order, resp_order),
    m_ylm_integral_norms(top_order), m_zonal_transformer(top_order),
    m_rotated_grid(top_order), m_rotated_exp(top_order)
{
    for (std::size_t l = 0; l  < top_order; ++l)
        m_ylm_integral_norms[l] = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

void AngleIntegratorCore<DistType::aniso, RespType::aniso>::resize(
    std::size_t geg_order, std::size_t resp_order, std::size_t top_order)
{
    m_rotor.expand(std::max(geg_order, resp_order));
    m_glq_transformer.resize(top_order);
    m_rotated_response_exp.reshape(resp_order);
    m_rotated_response_grid.reshape(top_order);

    m_aff_leg_integrals.resize(geg_order, resp_order);
    m_aff_leg_ylm_integrals.reshape(geg_order, resp_order);
    m_ylm_integral_norms.resize(top_order);
    for (std::size_t l = 0; l  < top_order; ++l)
        m_ylm_integral_norms[l] = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));

    m_zonal_transformer.resize(top_order);
    m_rotated_grid.reshape(top_order);
    m_rotated_exp.resize(top_order);
}

[[nodiscard]] double
AngleIntegratorCore<DistType::aniso, RespType::aniso>::integrate(
    zest::st::SphereGLQGridVectorSpan<const double> rotated_geg_zernike_grids,
    SHSpan<const double> response_exp,
    const la::Vector<double, 3>& offset, double rotation_angle, double shell,
    const zest::WignerdPiHalfCollection& wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);

    if (std::fabs(shell) > 1.0 + offset_len) return 0.0;

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);

    m_glq_transformer.backward_transform(m_rotated_response_exp, m_rotated_response_grid);

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    double res = 0.0;
    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        std::ranges::copy(m_rotated_response_grid.flatten(), m_rotated_grid.flatten().begin());
        util::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(m_rotated_grid, m_rotated_exp);
        res += util::inner_product(std::span<double>(m_rotated_exp), aff_leg_ylm_integrals_n.flatten());
    }

    return (2.0*std::numbers::pi)*res;
}

[[nodiscard]] std::array<double, 2>
AngleIntegratorCore<DistType::aniso, RespType::aniso>::integrate_transverse(
    zest::st::SphereGLQGridVectorSpan<const double> rotated_geg_zernike_grids,
    zest::st::SphereGLQGridVectorSpan<const double> rotated_trans_geg_zernike_grids,
    SHSpan<const double> response_exp,
    const la::Vector<double, 3>& offset, double rotation_angle, double shell,
    const zest::WignerdPiHalfCollection& wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    if (std::fabs(shell) > 1.0 + offset_len) return {0.0, 0.0};

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);

    m_glq_transformer.backward_transform(m_rotated_response_exp, m_rotated_response_grid);

    evaluate_aff_leg_ylm_integrals(shell, offset_len);

    //transverse contribution
    double trans = 0.0;
    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        std::ranges::copy(m_rotated_response_grid.flatten(), m_rotated_grid.flatten().begin());
        assert(m_rotated_grid.flatten().size() == rotated_trans_geg_zernike_grids[n].flatten().size());
        util::mul(m_rotated_grid.flatten(), rotated_trans_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(m_rotated_grid, m_rotated_exp);
        trans += util::inner_product(std::span<double>(m_rotated_exp), aff_leg_ylm_integrals_n.flatten());
    }

    TrapezoidSpan<const double>
    non_trans_aff_leg_ylm_integrals(
            m_aff_leg_ylm_integrals.data(),
            m_aff_leg_ylm_integrals.order() - 2,
            m_aff_leg_ylm_integrals.shape().extra_extent());

    // nontransverse contribution
    double nontrans = 0.0;
    for (std::size_t n : non_trans_aff_leg_ylm_integrals.indices())
    {
        auto non_trans_aff_leg_ylm_integrals_n = non_trans_aff_leg_ylm_integrals[n];
        std::ranges::copy(m_rotated_response_grid.flatten(), m_rotated_grid.flatten().begin());
        assert(m_rotated_grid.flatten().size() == rotated_geg_zernike_grids[n].flatten().size());
        util::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(m_rotated_grid, m_rotated_exp);
        nontrans += util::inner_product(std::span<const double>(m_rotated_exp), non_trans_aff_leg_ylm_integrals_n.flatten());
    }

    const double shell_sq = shell*shell;
    return {2.0*std::numbers::pi*nontrans, 2.0*std::numbers::pi*(trans - shell_sq*nontrans)};
}

void AngleIntegratorCore<DistType::aniso, RespType::aniso>::evaluate_aff_leg_ylm_integrals(
    double shell, double offset_len)
{
    m_aff_leg_integrals.integrals(m_aff_leg_ylm_integrals, shell, offset_len);

    for (std::size_t n : m_aff_leg_ylm_integrals.indices())
    {
        auto aff_leg_ylm_integrals_n = m_aff_leg_ylm_integrals[n];
        util::mul(aff_leg_ylm_integrals_n.flatten(), std::span(m_ylm_integral_norms));
    }
}

} // namespace zdm::zebra::detail
