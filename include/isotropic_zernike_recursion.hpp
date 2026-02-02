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

#include <zest/sequence.hpp>
#include <zest/shape.hpp>
#include <zest/shaped_array.hpp>

#include "utility.hpp"

namespace zdm::zebra
{

class IsotropicZernikeRecursion
{
public:
    IsotropicZernikeRecursion() = default;
    IsotropicZernikeRecursion(std::size_t max_order, std::size_t size):
        m_swap_chain{size}, m_r_sq(size), m_norms{max_order},
        m_k1{max_order}, m_k2{max_order}, m_k3{max_order}, m_max_order{max_order}
    {
        for (auto n : m_norms.indices())
        {
            m_norms[n] = std::sqrt(2.0*double(n) + 3.0);
            m_k1[n] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
            m_k2[n] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
            m_k3[n] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
        }
    }

    IsotropicZernikeRecursion(std::size_t max_order, std::span<const double> r):
        m_swap_chain{r.size()}, m_r_sq(r.size()), m_norms{max_order},
        m_k1{max_order}, m_k2{max_order}, m_k3{max_order}, m_max_order{max_order}
    {
        for (std::size_t i = 0; i < r.size(); ++i)
            m_r_sq[i] = r[i]*r[i];

        for (auto n : m_norms.indices())
        {
            m_norms[n] = std::sqrt(2.0*double(n) + 3.0);
            m_k1[n] = double(2*n - 1)*double(2*n + 1)/(double(n)*double(n + 1));
            m_k2[n] = -double(2*n - 1)*(1.0 + double(2*n + 1)*double(2*n - 3))/(double(n)*double(n + 1)*double(2*n - 3));
            m_k3[n] = -double(n - 2)*double(n - 1)*double(2*n + 1)/(double(n)*double(n + 1)*double(2*n - 3));
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return m_swap_chain.buffer_size(); }

    void resize(std::size_t size);
    void init(std::span<const double> x);

    template <std::regular_invocable<std::span<double>> Func>
    void init(const Func& f) noexcept
    {
        std::ranges::fill(m_swap_chain.current(), 1.0);
        f(m_r_sq);
        for (std::size_t i = 0; i < m_r_sq.size(); ++i)
            m_swap_chain.next()[i] = 2.5*m_r_sq[i] - 1.5;
        reset();
    }

    [[nodiscard]] std::span<const double>
    second_prev() const noexcept { return m_swap_chain.previous<2>(); }

    [[nodiscard]] std::span<const double>
    prev() const noexcept { return m_swap_chain.previous<1>(); }

    [[nodiscard]] std::span<const double>
    current() const noexcept { return m_swap_chain.current(); }

    void iterate() noexcept;
    void iterate(std::size_t n) noexcept;
    std::span<const double> next() noexcept;

private:
    void reset() noexcept;

    util::SwapChain<double, 3> m_swap_chain;
    std::vector<double> m_r_sq;
    zest::ShapedArray<double, zest::SequencedShape<zest::ParityLinearSequence>> m_norms;
    zest::ShapedArray<double, zest::SequencedShape<zest::ParityLinearSequence>> m_k1;
    zest::ShapedArray<double, zest::SequencedShape<zest::ParityLinearSequence>> m_k2;
    zest::ShapedArray<double, zest::SequencedShape<zest::ParityLinearSequence>> m_k3;
    std::size_t m_max_order{};
};

} // namespace zdm::zebra
