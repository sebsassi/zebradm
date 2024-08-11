#pragma once

#include "zest/zernike_glq_transformer.hpp"
#include "zest/sh_glq_transformer.hpp"
#include "zest/rotor.hpp"

#include "linalg.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"

namespace zebra
{
namespace parallel
{

class IsotropicAngleIntegrator
{
public:
    IsotropicAngleIntegrator() = default;
    explicit IsotropicAngleIntegrator(std::size_t num_threads);
    IsotropicAngleIntegrator(
        std::size_t dist_order, std::size_t num_threads);

    [[nodiscard]] std::size_t
    dist_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    num_threads() const noexcept { return m_num_threads; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball.

        @param distribution Zernike expansion of the distribution.
        @param boosts array of velocity boost vectors.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param out output values.
    */
    void integrate(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, zest::MDSpan<double, 2> out);
    
private:
    struct ThreadContext
    {
        zest::Rotor rotor;
        detail::IsotropicAngleIntegratorCore integrator;
    };

    [[nodiscard]] static constexpr std::size_t zernike_exp_size(
        std::size_t order) noexcept
    {
        return zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>::size(order);
    }

    [[nodiscard]] zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>
    accesss_rotated_geg_zernike_exp(
        std::size_t thread_id) noexcept;

    void integrate(
        std::size_t thread_id, const Vector<double, 3>& boost,
        std::span<const double> min_speeds, std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    std::vector<std::array<double, 2>> m_rotated_geg_zernike_exp;
    std::vector<ThreadContext> m_contexts;
    std::size_t m_dist_order;
    std::size_t m_num_threads;
};

class AnisotropicAngleIntegrator
{
public:
    AnisotropicAngleIntegrator() = default;
    AnisotropicAngleIntegrator(
        std::size_t num_teams, std::size_t threads_per_team);
    AnisotropicAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order, std::size_t num_teams,
        std::size_t threads_per_team);

    [[nodiscard]] std::size_t
    dist_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    resp_order() const noexcept { return m_resp_order; }

    [[nodiscard]] std::size_t
    trunc_order() const noexcept { return m_trunc_order; }

    [[nodiscard]] std::size_t
    num_teams() const noexcept { return m_num_teams; }

    [[nodiscard]] std::size_t
    threads_per_team() const noexcept { return m_threads_per_team; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a velocity disitribution on a boosted unit ball, combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution.
        @param boosts array of velocity boost vectors.
        @param min_speeds minimum speed parameters of the Radon transform.
        @param response spherical harmonic expansions of response at `min_speeds`.
        @param era Earth rotation angles.
        @param out output values.
        @param trunc_order maximum expansion order considered. See notes.

        @note This function assumes that the expansions in `distribution` and `response` have been defined in coordinate systems whose z-axes are aligned, and only differ by the angles expressed in `era`.

        @note Given two spherical harmonic expansions of orders `L` and `K`, the product expansion is of order `K + L`. Therefore, to avoid aliasing, computation of the Radon transform internally involves expansions of orders higher than that of `distribution`. However, if the expansion converges rapidly, the aliasing might be insignificant. The parameter `trunc_order` caps the order of any expansion used during the computation. This can significantly speed up the computation with some loss of accuracy.
    */
    void integrate(
        zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> distribution, std::span<const Vector<double, 3>> boosts, std::span<const double> min_speeds, SHExpansionCollectionSpan<const std::array<double, 2>> response, std::span<const double> era, zest::MDSpan<double, 2> out, std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    [[nodiscard]] static constexpr std::size_t zernike_exp_size(
        std::size_t order) noexcept
    {
        return zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>::size(order);
    }

    [[nodiscard]] static constexpr std::size_t sh_grid_size(
        std::size_t order) noexcept
    {
        return zest::st::SphereGLQGridSpan<double>::size(order);
    }

    [[nodiscard]] zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>>::SubSpan
    accesss_rotated_geg_zernike_exp(
        std::size_t team_id, std::size_t order) noexcept;

    [[nodiscard]] SuperSpan<zest::st::SphereGLQGridSpan<double>> 
    accesss_rotated_geg_zernike_exp_grids(std::size_t team_id);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    std::vector<zest::Rotor> m_rotors;
    zest::zt::ZernikeExpansionOrthoGeo m_geg_zernike_exp;
    std::vector<std::array<double, 2>> m_rotated_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_grids;
    std::vector<detail::AnisotropicAngleIntegratorCore> m_integrators;
    std::size_t m_dist_order;
    std::size_t m_resp_order;
    std::size_t m_trunc_order;
    std::size_t m_num_teams;
    std::size_t m_threads_per_team;
};

} // namespace parallel
} // namespace zebra