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

#include "polynomial.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"

namespace zdm
{

template <DistType dist_type, RespType resp_type>
class ElectronRateCalculator {};

template <>
class ElectronRateCalculator<DistType::iso, RespType::iso>
{
public:
    void rate(
        IsotropicZernikeVectorSpan<const double> target_response, std::span<la::Vector<double, 3>> lab_velocities,
        std::span<double> energies, double max_momentum_transfer, double max_speed, double dm_mass,
        zest::DynamicMDSpan<double, 2> out)
    {
    }

    void partial_double_differential_rate(
        IsotropicZernikeVectorSpan<const double> target_response, std::span<la::Vector<double, 3>> lab_velocities,
        std::span<double> energies, double max_momentum_transfer, double max_speed, double dm_mass,
        zest::DynamicMDSpan<double, 2> out)
    {
        zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::cos>(
                m_shell_glq_nodes, m_shell_glq_weights, m_shell_glq_nodes.size() & 1);

        for (std::size_t i = 0; i < lab_velocities.size(); ++i)
        {
            const double lab_speed = la::length(lab_velocities[i]);
            const double offset_len = lab_speed/max_speed;
            for (std::size_t j = 0; j < energies.size(); ++j)
            {
                // Kinematically forbidden; bail out
                if (energies[j] > 0.5*dm_mass*(max_speed + lab_speed)*(max_speed + lab_speed))
                    continue;

                if (energies[j] > 0.5*dm_mass*(max_speed - lab_speed)*(max_speed - lab_speed))
                    /* TODO single segment */;
                else
                    /* TODO three segments */;

                for (std::size_t k = 0; k < m_shells.size(); ++k)
                    m_partial_dd_rate[i, j, k] = m_target_response_shells[k]*m_angle_integrator.integrate(m_dist_radon, offset_len, m_shells[k]);
            }
        }
    }

    template <std::size_t N>
    void differential_rate(const Polynomial<double, N>& form_factor, zest::DynamicMDSpan<double, 2> out)
    {
        for (std::size_t i = 0; i < m_partial_dd_rate.extent(0); ++i)
        {
            for (std::size_t j = 0; j < m_partial_dd_rate.extent(1); ++j)
            {
                for (std::size_t k = 0; k < m_partial_dd_rate.extent(2); ++k)
                {
                    const double momentum_transfer_sq = m_momentum_transfers[i, j, k]*m_momentum_transfers[i, j, k];
                    out[i, j] += m_shell_weights[k]*form_factor(momentum_transfer_sq)*m_momentum_transfers[i, j, k]*m_partial_dd_rate[i, j, k];
                }
            }
        }
    }

private:
    std::vector<double> m_target_response_shells;
    ZernikeExpansion<double> m_dist_radon;
    std::vector<double> m_shell_glq_nodes;
    std::vector<double> m_shell_glq_weights;
    std::vector<double> m_shells;
    zest::DynamicMDSpan<double, 3> m_momentum_transfers;
    zest::DynamicMDArray<double, 3> m_partial_dd_rate;
    zebra::detail::AngleIntegratorCore<DistType::iso, RespType::iso> m_angle_integrator;
};


} // namespace zdm
