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
#include "zebra_radon.hpp"
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

[[nodiscard]] constexpr QuantityOf<reduced_mass> auto
reduced_mass(QuantityOf<mass> auto m1, QuantityOf<mass> auto m2) noexcept
{
    return m1*m2/(m1 + m2);
}

template <DistType dist_type, RespType resp_type>
class ElectronRateCalculator {};

template <>
class ElectronRateCalculator<DistType::iso, RespType::iso>
{
public:

    template <
        QuantityOf<velocity> Velocity,
        QuantityOf<energy> Energy,
        QuantityOf<energy_differential_rate_per_unit_mass> Rate
    >
    [[nodiscard]] zest::DynamicMDSpan<Rate, 2> energy_differential_rate(
        IsotropicZernikeVectorSpan<const double> velocity_distribution,
        QuantityOf<speed> auto max_speed,
        QuantityOf<mass> auto dm_mass,
        QuantityOf<energy_density> auto dm_energy_density,
        QuantityOf<cross_section> auto dm_electron_cross_section,
        IsotropicZernikeVectorSpan<const double> target_response,
        QuantityOf<momentum_transfer> auto max_momentum_transfer,
        QuantityOf<mass_density> auto target_density,
        std::span<const Velocity> lab_velocities,
        std::span<const Energy> energies,
        zest::DynamicMDSpan<Rate, 2> out)
    {
        const quantity inv_max_speed = 1.0/max_speed;

        // All physical units should be in the prefactor.
        const quantity prefactor = scattering_rate_prefactor(
                dm_mass, dm_energy_density, target_density, dm_electron_cross_section,
                max_momentum_transfer, max_speed);

        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::cos>(
                m_shell_glq_nodes, m_shell_glq_weights, m_shell_glq_nodes.size() & 1);
        const std::size_t num_nodes = m_shell_glq_nodes.size();

        zebra::radon_transform(velocity_distribution);
        for (std::size_t i = 0; i < lab_velocities.size(); ++i)
        {
            const quantity lab_speed = lab_velocities[i].magnitude();
            generate_optimal_momentum_grid(
                    lab_speed, energies, max_speed, max_momentum_transfer, dm_mass);
            calculate_shells(energies, dm_mass, max_speed);
            m_angle_integrator.integrate(
                    static_cast<double>(lab_speed*inv_max_speed),
                    m_shell_grid.flatten(), m_aiwrt_grid.flatten());
            m_grid_evaluator.evaluate(
                    target_response,
                    m_normalized_momentum_grid.flatten(), m_response_grid.flatten());

            util::mul(m_aiwrt_grid.flatten(), m_response_grid.flatten());

            for (std::size_t j = 0; j < energies.size(); ++j)
            {
                std::span<double> aiwrt = m_aiwrt_grid[j];
                std::span<const double> interval_weights = m_interval_weights[j];

                const std::size_t num_intervals = aiwrt.size()/num_nodes;
                for (std::size_t k = 0; k < num_intervals; ++k)
                {
                    for (std::size_t l = 0; l < num_nodes; ++l)
                        aiwrt[num_nodes*k + l] *= interval_weights[k]*m_shell_glq_weights[l];
                }
            }

            util::mul(m_aiwrt_grid.flatten(), m_normalized_momentum_grid.flatten());

            for (std::size_t j = 0; j < energies.size(); ++j)
                out[i, j] = (prefactor*util::sum(m_aiwrt_grid[j])).in(Rate::unit);
        }
    }

    template <QuantityOf<energy> Energy>
    void generate_optimal_momentum_grid(
        QuantityOf<speed> auto lab_speed,
        std::span<const Energy> energies,
        QuantityOf<speed> auto max_speed,
        QuantityOf<momentum_transfer> auto max_momentum_transfer,
        QuantityOf<mass> auto dm_mass)
    {
        m_shell_grid.clear();

        const quantity inv_max_momentum = 1.0/max_momentum_transfer;
        const quantity speed_lo = max_speed - lab_speed;
        const quantity speed_hi = max_speed + lab_speed;
        const quantity speed_lo_sq = speed_lo*speed_lo;
        const quantity speed_hi_sq = speed_hi*speed_hi;
        const quantity emax_lo = 0.5*dm_mass*speed_lo_sq;
        const quantity emax_hi = 0.5*dm_mass*speed_hi_sq;

        const auto normalized_momentum_lo = static_cast<double>(inv_max_momentum*dm_mass*speed_lo);
        const auto normalized_momentum_hi = static_cast<double>(inv_max_momentum*dm_mass*speed_hi);

        const double normalized_momentum_lo_sq = normalized_momentum_lo*normalized_momentum_lo;
        const double normalized_momentum_hi_sq = normalized_momentum_hi*normalized_momentum_hi;

        for (std::size_t i = 0; i < energies.size(); ++i)
        {
            const auto normalized_mom_sq_floor
                = static_cast<double>((2.0*inv_max_momentum*inv_max_momentum*dm_mass)*energies[i]);
            // Kinematically forbidden; bail out
            if (energies[i] > emax_hi)
            {
                m_normalized_momentum_grid.append(0);
                m_interval_weights.append(0);
            }
            else if (energies[i] > emax_lo)
            {
                const double normalized_momentum_hi_min 
                    = normalized_momentum_hi
                        - std::sqrt(normalized_momentum_hi_sq - normalized_mom_sq_floor);
                if (1.0 < normalized_momentum_hi_min)
                {
                    m_normalized_momentum_grid.append(0);
                    m_interval_weights.append(0);
                    continue;
                }

                const double normalized_momentum_hi_max
                    = normalized_momentum_hi
                        + std::sqrt(normalized_momentum_hi_sq - normalized_mom_sq_floor);

                const std::array intervals = {
                    std::array<double, 2>{
                        normalized_momentum_hi_min,
                        std::min(normalized_momentum_hi_max, 1.0)
                    }
                };
                std::span<double> momenta = m_normalized_momentum_grid.append(m_shell_glq_nodes.size());
                generate_momenta_on(intervals, momenta);
                m_interval_weights.append(weights_from_intervals(intervals));
            }
            else
            {
                const double normalized_momentum_hi_min
                    = normalized_momentum_hi
                        - std::sqrt(normalized_momentum_hi_sq - normalized_mom_sq_floor);
                if (1.0 < normalized_momentum_hi_min)
                {
                    m_shell_grid.append(0);
                    m_interval_weights.append(0);
                    continue;
                }

                const double normalized_momentum_lo_min
                    = normalized_momentum_lo
                        - std::sqrt(normalized_momentum_lo_sq - normalized_mom_sq_floor);
                if (1.0 < normalized_momentum_lo_min)
                {
                    const std::array intervals = {
                        std::array<double, 2>{
                            normalized_momentum_hi_min,
                            1.0
                        }
                    };
                    std::span<double> normalized_momenta = m_normalized_momentum_grid.append(m_shell_glq_nodes.size());
                    generate_momenta_on(intervals, normalized_momenta);
                    m_interval_weights.append(weights_from_intervals(intervals));
                    continue;
                }

                const double normalized_momentum_lo_max
                    = normalized_momentum_lo
                        + std::sqrt(normalized_momentum_lo_sq - normalized_mom_sq_floor);
                if (1.0 < normalized_momentum_lo_max)
                {
                    const std::array intervals = {
                        std::array<double, 2>{
                            normalized_momentum_hi_min,
                            normalized_momentum_lo_min
                        },
                        std::array<double, 2>{
                            normalized_momentum_lo_min,
                            1.0,
                        }
                    };
                    std::span<double> normalized_momenta = m_normalized_momentum_grid.append(2*m_shell_glq_nodes.size());
                    generate_momenta_on(intervals, normalized_momenta);
                    m_interval_weights.append(weights_from_intervals(intervals));
                    continue;
                }

                const double normalized_momentum_hi_max
                    = normalized_momentum_hi
                        + std::sqrt(normalized_momentum_hi_sq - normalized_mom_sq_floor);

                const std::array intervals = {
                    std::array<double, 2>{
                        normalized_momentum_hi_min,
                        normalized_momentum_lo_min
                    },
                    std::array<double, 2>{
                        normalized_momentum_lo_min,
                        normalized_momentum_lo_max,
                    },
                    std::array<double, 2>{
                        normalized_momentum_lo_max,
                        std::min(normalized_momentum_hi_max, 1.0)
                    }
                };
                std::span<double> normalized_momenta = m_normalized_momentum_grid.append(3*m_shell_glq_nodes.size());
                generate_momenta_on(intervals, normalized_momenta);
                m_interval_weights.append(weights_from_intervals(intervals));
            }

        }

    }

private:
    [[nodiscard]] static constexpr QuantityOf<energy_differential_rate_per_unit_mass> auto
    scattering_rate_prefactor(
        QuantityOf<mass> auto dm_mass,
        QuantityOf<energy_density> auto dm_energy_density,
        QuantityOf<mass_density> auto target_density,
        QuantityOf<cross_section> auto dm_electron_cross_section,
        QuantityOf<momentum_transfer> auto max_momentum_transfer,
        QuantityOf<speed> auto max_speed) noexcept
    {
        constexpr double two_pi = 2.0*std::numbers::pi;
        constexpr double two_pi_cubed = two_pi*two_pi*two_pi;
        const quantity red_mass = reduced_mass(dm_mass, (1.0*electron_mass).in(dm_mass.unit));
        const quantity numerator = (std::numbers::pi/two_pi_cubed)*dm_energy_density*dm_electron_cross_section*max_momentum_transfer*max_momentum_transfer;
        const quantity denominator = target_density*dm_mass*red_mass*red_mass;
        return numerator/denominator;
    }

    template <std::size_t N>
    void generate_momenta_on(
        const std::array<std::array<double, 2>, N>& momentum_intervals,
        std::span<double> normalized_momenta) noexcept
    {
        const std::size_t num_nodes = m_shell_glq_nodes.size();
        assert(normalized_momenta.size() == N*num_nodes);

        for (std::size_t i = 0; i < N; ++i)
        {
            std::span<double> interval_momenta = normalized_momenta.subspan(i*num_nodes, num_nodes);
            const double mid_point = 0.5*(momentum_intervals[i][1] + momentum_intervals[i][0]);
            const double half_width = 0.5*(momentum_intervals[i][1] - momentum_intervals[i][0]);
            for (std::size_t j = 0; j < num_nodes; ++j)
            {
                const double momentum = half_width*m_shell_glq_nodes[j] + mid_point;
                interval_momenta[j] = momentum;
            }
        }
    }

    template <QuantityOf<energy> Energy>
    void calculate_shells(
        std::span<const Energy> energies,
        QuantityOf<mass> auto dm_mass,
        QuantityOf<speed> auto max_speed,
        QuantityOf<momentum_transfer> auto max_momentum_transfer) noexcept
    {
        const quantity inverted_half_mass = 0.5/dm_mass;
        const quantity mass_factor = max_momentum_transfer/max_speed;
        for (std::size_t i = 0; i < energies.size(); ++i)
        {
            std::span<const double> normalized_momenta = m_normalized_momentum_grid[i];
            std::span<double> shells = m_shell_grid[i];
            for (std::size_t j = 0; j < normalized_momenta.size(); ++i)
                shells[j] = static_cast<double>(
                        (normalized_momenta[j]*inverted_half_mass
                            + energies[i]/normalized_momenta[j])*mass_factor);
        }
    }

    template <typename T, std::size_t N>
    [[nodiscard]] static constexpr std::array<T, N>
    weights_from_intervals(const std::array<std::array<T, 2>, N>& intervals) noexcept
    {
        return [&]<std::size_t... I>(
            const std::array<std::array<T, 2>, N>& intervals, std::index_sequence<I...>)
        {
            return std::array<T, N>{0.5*(intervals[I][1] - intervals[I][0])...};
        }(intervals, std::make_index_sequence<N>{});
    }

    std::vector<double> m_shell_glq_nodes;
    std::vector<double> m_shell_glq_weights;
    RaggedTable<double> m_normalized_momentum_grid;
    RaggedTable<double> m_shell_grid;
    RaggedTable<double> m_aiwrt_grid;
    RaggedTable<double> m_response_grid;
    RaggedTable<double> m_interval_weights;
    zebra::AngleIntegrator<DistType::iso, RespType::iso> m_angle_integrator;
};


} // namespace zdm
