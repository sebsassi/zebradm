/*
Copyright (c) 2026 Sebastian Sassi

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

#include <algorithm>

#include "polynomial.hpp"
#include "types.hpp"
#include "constants.hpp"
#include "zebra_angle_integrator.hpp"

namespace zdm
{

template <typename T>
class RaggedTable
{
public:
    using size_type = std::size_t;

    RaggedTable() = default;
    explicit RaggedTable(std::span<size_type> sizes):
        m_data(std::ranges::fold_left(sizes, 0, std::plus{})),
        m_offsets(sizes.size() + 1)
    {
        size_type offset = 0;
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            m_offsets[i] = offset;
            offset += sizes[i];
        }
        m_offsets.back() = offset;
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_data.size(); }

    void clear()
    {
        m_data.clear();
        m_offsets.clear();
    }

    std::span<T> append(size_type size)
    {
        m_data.resize(m_data.size() + size);
        m_offsets.emplace_back(m_data.size());
    }

    std::span<T> append(std::span<const T> row)
    {
        m_data.append_range(row);
        m_offsets.emplace_back(m_data.size());
    }

    [[nodiscard]] std::span<T> flatten() noexcept
    {
        return m_data;
    }

    [[nodiscard]] std::span<const T> flatten() const noexcept
    {
        return m_data;
    }

    [[nodiscard]] std::span<T> front() noexcept
    {
        return {m_data.data(), m_offsets[1] - m_offsets.front()};
    }

    [[nodiscard]] std::span<const T> front() const noexcept
    {
        return {m_data.data(), m_offsets[1] - m_offsets.front()};
    }

    [[nodiscard]] std::span<T> back() noexcept
    {
        return {m_data.data() + m_offsets[m_data.size() - 1], m_offsets.back() - m_offsets[m_data.size() - 1]};
    }

    [[nodiscard]] std::span<const T> back() const noexcept
    {
        return {m_data.data() + m_offsets[m_data.size() - 1], m_offsets.back() - m_offsets[m_data.size() - 1]};
    }

    [[nodiscard]] std::span<T> operator[](size_type i) noexcept
    {
        return {m_data.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }

    [[nodiscard]] std::span<const T> operator[](size_type i) const noexcept
    {
        return {m_data.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }

private:
    std::vector<T> m_data;
    std::vector<size_type> m_offsets;
};

[[nodiscard]] constexpr double reduced_mass(double m1, double m2) noexcept
{
    return m1*m2/(m1 + m2);
}

template <DistType dist_type, RespType resp_type>
class ElectronRateCalculator {};

template <>
class ElectronRateCalculator<DistType::iso, RespType::iso>
{
public:

    void differential_rate(
        IsotropicZernikeVectorSpan<const double> velocity_distribution,
        IsotropicZernikeVectorSpan<const double> target_response,
        std::span<la::Vector<double, 3>> lab_velocities, std::span<double> energies,
        double max_momentum_transfer, double max_speed, double dm_mass, double dm_energy_density,
        double dm_electron_cross_section, double target_density,
        zest::DynamicMDSpan<double, 2> out)
    {
        const double inv_max_speed = 1.0/max_speed;
        const double prefactor = scattering_rate_prefactor(
                dm_mass, dm_energy_density, target_density, dm_electron_cross_section);

        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::cos>(
                m_shell_glq_nodes, m_shell_glq_weights, m_shell_glq_nodes.size() & 1);
        const std::size_t num_nodes = m_shell_glq_nodes.size();

        m_angle_integrator.radon_transform(velocity_distribution);
        for (std::size_t i = 0; i < lab_velocities.size(); ++i)
        {
            const double lab_speed = lab_velocities[i].magnitude();
            generate_optimal_momentum_grid(lab_speed, energies, max_speed, max_momentum_transfer, dm_mass);
            calculate_shells(energies, dm_mass, max_speed);
            m_angle_integrator.integrate(lab_speed*inv_max_speed, m_shell_grid.flatten(), m_aiwrt_grid.flatten());
            m_grid_evaluator.evaluate(target_response, m_momentum_grid.flatten(), m_response_grid.flatten());

            for (std::size_t j = 0; j < energies.size(); ++j)
            {
                const std::span<const double> aiwrt = m_aiwrt_grid[j];
                std::span<double> results = m_result_grid[j];
                const std::size_t num_intervals = m_shell_grid.size()/num_nodes;
                for (std::size_t k = 0; k < num_intervals; ++k)
                {
                    for (std::size_t l = 0; l < num_nodes; ++l)
                        results[num_nodes*k + l] = m_shell_glq_weights[l]*aiwrt[num_nodes*k + l];
                }
            }

            util::mul(m_result_grid.flatten(), m_response_grid.flatten());
            util::mul(m_result_grid.flatten(), m_momentum_grid.flatten());

            for (std::size_t j = 0; j < energies.size(); ++j)
                out[i, j] = prefactor*util::sum(m_result_grid[j]);
        }
    }

    void generate_optimal_momentum_grid(
        double lab_speed, std::span<double> energies,
        double max_speed, double max_momentum_transfer, double dm_mass)
    {
        m_shell_grid.clear();

        const double speed_lo = max_speed - lab_speed;
        const double speed_hi = max_speed + lab_speed;
        const double speed_lo_sq = speed_lo*speed_lo;
        const double speed_hi_sq = speed_hi*speed_hi;
        const double momentum_lo = dm_mass*speed_lo;
        const double momentum_hi = dm_mass*speed_hi;
        const double emax_lo = 0.5*dm_mass*speed_lo_sq;
        const double emax_hi = 0.5*dm_mass*speed_hi_sq;

        for (std::size_t i = 0; i < energies.size(); ++i)
        {
            // Kinematically forbidden; bail out
            if (energies[i] > emax_hi)
            {
                m_momentum_grid.append(0);
            }
            else if (energies[i] > emax_lo)
            {
                const double momentum_hi_min 
                    = momentum_hi - std::sqrt(momentum_hi*momentum_hi - 2.0*dm_mass*energies[1]);
                if (max_momentum_transfer < momentum_hi_min)
                {
                    m_momentum_grid.append(0);
                    continue;
                }

                const double momentum_hi_max
                    = momentum_hi + std::sqrt(momentum_hi*momentum_hi - 2.0*dm_mass*energies[1]);

                std::span<double> momenta = m_momentum_grid.append(m_shell_glq_nodes.size());
                generate_momenta_on(
                    {
                        std::array<double, 2>{
                            momentum_hi_min,
                            std::min(momentum_hi_max, max_momentum_transfer)
                        }
                    },
                    momenta);
            }
            else
            {
                const double momentum_hi_min
                    = momentum_hi - std::sqrt(momentum_hi*momentum_hi - 2.0*dm_mass*energies[i]);
                if (max_momentum_transfer < momentum_hi_min)
                {
                    m_shell_grid.append(0);
                    continue;
                }

                const double momentum_lo_min
                    = momentum_lo - std::sqrt(momentum_lo*momentum_lo - 2.0*dm_mass*energies[i]);
                if (max_momentum_transfer < momentum_lo_min)
                {
                    std::span<double> momenta = m_momentum_grid.append(m_shell_glq_nodes.size());
                    generate_momenta_on(
                        {
                            std::array<double, 2>{
                                momentum_hi_min,
                                max_momentum_transfer
                            }
                        },
                        momenta);
                    continue;
                }

                const double momentum_lo_max
                    = momentum_lo + std::sqrt(momentum_lo*momentum_lo - 2.0*dm_mass*energies[i]);
                if (max_momentum_transfer < momentum_lo_max)
                {
                    std::span<double> momenta = m_momentum_grid.append(2*m_shell_glq_nodes.size());
                    generate_momenta_on(
                        {
                            std::array<double, 2>{
                                momentum_hi_min,
                                momentum_lo_min
                            },
                            std::array<double, 2>{
                                momentum_lo_min,
                                max_momentum_transfer,
                            }
                        },
                        momenta);
                    continue;
                }

                const double momentum_hi_max
                    = momentum_hi + std::sqrt(momentum_hi*momentum_hi - 2.0*dm_mass*energies[i]);

                std::span<double> momenta = m_momentum_grid.append(3*m_shell_glq_nodes.size());
                generate_momenta_on(
                    {
                        std::array<double, 2>{
                            momentum_hi_min,
                            momentum_lo_min
                        },
                        std::array<double, 2>{
                            momentum_lo_min,
                            momentum_lo_max,
                        },
                        std::array<double, 2>{
                            momentum_lo_max,
                            std::min(momentum_hi_max, max_momentum_transfer)
                        }
                    },
                    momenta);
            }

        }

    }

private:
    [[nodiscard]] static constexpr double scattering_rate_prefactor(
        double dm_mass, double dm_energy_density, double target_density,
        double dm_electron_cross_section) noexcept
    {
        constexpr double two_pi = 2.0*std::numbers::pi;
        constexpr double two_pi_cubed = two_pi*two_pi*two_pi;
        const double red_mass = reduced_mass(dm_mass, constants::electron_mass<units::GeV>);
        const double numerator = (std::numbers::pi/two_pi_cubed)*dm_energy_density*dm_electron_cross_section;
        const double denominator = target_density*dm_mass*red_mass*red_mass;
        return numerator/denominator;
    }

    template <std::size_t N>
    void generate_momenta_on(
        const std::array<std::array<double, 2>, N>& momentum_intervals,
        std::span<double> momenta) noexcept
    {
        const std::size_t num_nodes = m_shell_glq_nodes.size();
        assert(momenta.size() == N*num_nodes);

        for (std::size_t i = 0; i < N; ++i)
        {
            std::span<double> interval_momenta = momenta.subspan(i*num_nodes, num_nodes);
            const double mid_point = 0.5*(momentum_intervals[i][1] + momentum_intervals[i][0]);
            const double half_width = 0.5*(momentum_intervals[i][1] - momentum_intervals[i][0]);
            for (std::size_t j = 0; j < num_nodes; ++j)
            {
                const double momentum = half_width*m_shell_glq_nodes[j] + mid_point;
                interval_momenta[j] = momentum;
            }
        }
    }

    void calculate_shells(
        std::span<const double> energies, double dm_mass, double max_speed) noexcept
    {
        const double mass_factor = 0.5/dm_mass;
        const double inv_max_speed = 1.0/max_speed;
        for (std::size_t i = 0; i < energies.size(); ++i)
        {
            std::span<const double> momenta = m_momentum_grid[i];
            std::span<double> shells = m_shell_grid[i];
            for (std::size_t j = 0; j < momenta.size(); ++i)
                shells[j] = (momenta[j]*mass_factor + energies[i]/momenta[j])*inv_max_speed;
        }
    }

    std::vector<double> m_shell_glq_nodes;
    std::vector<double> m_shell_glq_weights;
    RaggedTable<double> m_momentum_grid;
    RaggedTable<double> m_shell_grid;
    RaggedTable<double> m_aiwrt_grid;
    RaggedTable<double> m_response_grid;
    RaggedTable<double> m_result_grid;
    zebra::AngleIntegrator<DistType::iso, RespType::iso> m_angle_integrator;
};


} // namespace zdm
