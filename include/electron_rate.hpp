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
#include <span>
#include <vector>

#include "zest/md_span.hpp"

#include "types.hpp"
#include "vector.hpp"

template <typename T>
class RaggedTable
{
    RaggedTable() = default;
    RaggedTable(std::span<std::size_t> sizes):
        m_data(std::ranges::fold_left_first(sizes, std::plus<std::size_t>{})),
        m_offsets(sizes.size() + 1)
    {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            m_offsets[i] = offset;
            offset += sizes[i];
        }
        m_offsets.back() = offset;
    }

    void resize(std::span<std::size_t> sizes)
    {
        m_data.resize(std::ranges::fold_left_first(sizes, std::plus<std::size_t>{}));
        m_offsets.resize(sizes.size() + 1);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            m_offsets[i] = offset;
            offset += sizes[i];
        }
        m_offsets.back() = offset;
    }

    void append_row(std::span<const T> row)
    {
        m_data.append_range(row);
        m_offsets.push_back(m_data.size());
    }

    void append_row(std::size_t size)
    {
        m_data.resize(m_data.size() + size);
        m_offsets.push_back(m_data.size());
    }

    [[nodiscard]] std::span<const T>
    operator[](std::size_t i) const noexcept
    {
        return {m_data.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }

    [[nodiscard]] std::span<T>
    operator[](std::size_t i) noexcept
    {
        return {m_data.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }

private:
    std::vector<T> m_data;
    std::vector<std::size_t> m_offsets;
};

namespace zdm::electron
{

template <DistType dist_type, RespType resp_type>
class RateCalculator;

template <>
class RateCalculator<zdm::DistType::iso, zdm::RespType::iso>
{
public:
    RateCalculator() = default;

    void event_rate(
        IsotropicZernikeSpan<double> velocity_distribution, double max_velocity,
        IsotropicZernikeVectorSpan<double> target_response, double max_momentum_transfer,
        std::span<la::Vector<double, 3>> lab_velocities,
        std::span<double> energies,
        zest::DynamicMDSpan<double, 2> out)
    {
        
    }

private:

};

} // namespace zdm::electron
