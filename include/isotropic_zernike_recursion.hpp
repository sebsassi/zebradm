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
#pragma once

#include <algorithm>

#include <zest/sequence.hpp>
#include <zest/shape.hpp>
#include <zest/shaped_array.hpp>
#include <zest/zernike_conventions.hpp>

#include "utility.hpp"

namespace zdm::zebra
{

template <zest::zt::ZernikeNorm zernike_norm_param>
class IsotropicRadialZernikeRecursion
{
public:
    static constexpr zest::zt::ZernikeNorm zernike_norm = zernike_norm_param;

    IsotropicRadialZernikeRecursion() = default;
    IsotropicRadialZernikeRecursion(std::size_t max_order, std::size_t size):
        m_swap_chain{size}, m_r_sq(size), m_k{max_order}
    {
        for (auto n : m_k.indices(4))
        {
            if constexpr (zernike_norm == zest::zt::ZernikeNorm::unnormed)
            {
                m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
            else
            {
                m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
        }
    }

    IsotropicRadialZernikeRecursion(std::size_t max_order, std::span<const double> r):
        m_swap_chain{r.size()}, m_r_sq(r.size()), m_k{max_order}
    {
        for (std::size_t i = 0; i < r.size(); ++i)
            m_r_sq[i] = r[i]*r[i];

        for (auto n : m_k.indices(4))
        {
            if constexpr (zernike_norm == zest::zt::ZernikeNorm::unnormed)
            {
                m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
            else
            {
                m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
            }
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_swap_chain.buffer_size(); }

    void expand(std::size_t max_order)
    {
        if (max_order > m_max_order)
        {
            m_k.reshape(max_order);
            for (auto n : m_k.indices(std::max(m_max_order, 4UL)))
            {
                if constexpr (zernike_norm == zest::zt::ZernikeNorm::unnormed)
                {
                    m_k[n, 0] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
                    m_k[n, 1] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                    m_k[n, 2] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
                }
                else
                {
                    m_k[n, 0] = std::sqrt(double(2*n + 3)*double(2*n - 1))*double(2*n + 1)/(double(n)*double(n + 1));
                    m_k[n, 1] = -std::sqrt(double(2*n + 3)*double(2*n - 1))*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
                    m_k[n, 2] = -std::sqrt(double(2*n + 3)/double(2*n - 5))*double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
                }
            }
        }
    }

    void resize(std::size_t max_order, std::size_t size)
    {
        expand(max_order);
        m_swap_chain.resize(size);
        m_r_sq.resize(size);
    }

    void init()
    {
        constexpr double sqrt7 = 0.0;
        constexpr double radial_zernike_0 = (zernike_norm == zest::zt::ZernikeNorm::unnormed) ?
            1.0 : std::numbers::sqrt3;
        std::ranges::fill(m_swap_chain.current(), radial_zernike_0);

        for (std::size_t i = 0; i < m_r_sq.size(); ++i)
        {
            if constexpr (zernike_norm == zest::zt::ZernikeNorm::unnormed)
                m_swap_chain.next()[i] = 2.5*m_r_sq[i] - 1.5;
            else
                m_swap_chain.next()[i] = (2.5*sqrt7)*m_r_sq[i] - 1.5*sqrt7;
        }
        reset();
    }

    void init(std::span<const double> x)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
            m_r_sq[i] = x[i]*x[i];
        init();
    }

    template <std::regular_invocable<std::span<double>> Func>
    void init(const Func& f) noexcept
    {
        f(m_r_sq);
        init();
    }

    [[nodiscard]] std::span<const double>
    second_prev() const noexcept { return m_swap_chain.previous<2>(); }

    [[nodiscard]] std::span<const double>
    prev() const noexcept { return m_swap_chain.previous<1>(); }

    [[nodiscard]] std::span<const double>
    current() const noexcept { return m_swap_chain.current(); }

    void iterate() noexcept
    {
        if (m_n + 2 > m_max_order) [[unlikely]]
            expand(m_max_order + (m_max_order >> 1));

        m_swap_chain.advance();

        if (m_n > 0) [[likely]]
        {
            const double k1 = m_k[m_n, 0];
            const double k2 = m_k[m_n, 1];
            const double k3 = m_k[m_n, 2];
            for (std::size_t i = 0; i < m_r_sq.size(); ++i)
                m_swap_chain.current()[i] = (k1*m_r_sq[i] + k2)*m_swap_chain.previous<1>()[i] - k3*m_swap_chain.previous<2>()[i];
        }
        ++m_n;

    }

    void iterate(std::size_t n) noexcept
    {
        if (n == 0) [[unlikely]] return;
        if (m_n + n + 1 > m_max_order) [[unlikely]]
            expand(m_max_order + (m_max_order >> 1));

        if (m_n == 0) [[unlikely]]
        {
            m_swap_chain.advance();
            ++m_n;
            --n;
        }

        for (std::size_t i = 0; i < n; ++i)
        {
            m_swap_chain.advance();

            const double k1 = m_k[m_n, 0];
            const double k2 = m_k[m_n, 1];
            const double k3 = m_k[m_n, 2];
            for (std::size_t i = 0; i < m_r_sq.size(); ++i)
                m_swap_chain.current()[i] = (k1*m_r_sq[i] + k2)*m_swap_chain.previous<1>()[i] - k3*m_swap_chain.previous<2>()[i];
            ++m_n;
        }

    }

    [[nodiscard]] std::span<const double> next() noexcept
    {
        iterate();
        return current();
    }

private:
    void reset() noexcept { m_n = 0; }

    util::SwapChain<double, 3> m_swap_chain;
    std::vector<double> m_r_sq;
    zest::ShapedArray<double, zest::TensorSequenceShape<zest::ParityLinearSequence, 3>> m_k;
    std::size_t m_n{};
    std::size_t m_max_order{};
};

} // namespace zdm::zebra
