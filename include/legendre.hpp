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
#include <concepts>
#include <span>
#include <vector>

#include <zest/md_array.hpp>
#include <zest/md_span.hpp>

#include "utility.hpp"

namespace zdm::zebra
{

void legendre_recursion_vec(
    zest::DynamicMDSpan<double, 2> legendre, std::span<double> x);

void legendre_recursion(std::span<double> legendre, double x);

template <typename T>
class LegendreRecursion
{
public:
    LegendreRecursion() = default;
    LegendreRecursion(T x): m_x{x} {}

    void init(T x)
    {
        m_x = x;
        m_second_prev = 0.0;
        m_prev = 0.0;
        m_current = 1.0;
        m_l = 0;
    }

    [[nodiscard]] T second_prev() const noexcept { return m_second_prev; }
    [[nodiscard]] T prev() const noexcept { return m_prev; }
    [[nodiscard]] T current() const noexcept { return m_current; }

    template <std::size_t N = 1>
    void iterate() noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            const double inv_l = 1.0/double(m_l + 1);
            const double a = double(2*m_l + 1)*inv_l;
            const double b = double(m_l)*inv_l;
            m_second_prev = m_prev;
            m_prev = m_current;
            m_current = a*m_x*m_prev + b*m_second_prev;
            ++m_l;
        }
    }

    void iterate(std::size_t n) noexcept
    {
        for (std::size_t i = 0; i < n; ++i)
            iterate<>();
    }

    template <std::size_t N = 1>
    [[nodiscard]] T next() noexcept
    {
        iterate<N>();
        return m_current;
    }

private:
    T m_x = 0.0;
    T m_second_prev = 0.0;
    T m_prev = 0.0;
    T m_current = 1.0;
    std::size_t m_l = 0;
};

class LegendreArrayRecursion
{
public:
    LegendreArrayRecursion() = default;
    explicit LegendreArrayRecursion(std::size_t size);
    explicit LegendreArrayRecursion(std::span<const double> x);

    [[nodiscard]] std::size_t size() const noexcept { return m_swap_chain.buffer_size(); }

    void resize(std::size_t size);
    void init(std::span<const double> x);

    template <std::regular_invocable<std::span<double>> Func>
    void init(const Func& f) noexcept
    {
        std::ranges::fill(m_swap_chain.current(), 1.0);
        f(m_x);
        std::ranges::copy(m_x, m_swap_chain.next().begin());
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
    [[nodiscard]] std::span<const double> next() noexcept;

private:
    void reset() noexcept;

    util::SwapChain<double, 3> m_swap_chain;
    std::array<std::vector<double>, 3> m_buffers;
    std::vector<double> m_x;
    std::size_t m_l = 0;
};

class LegendreIntegralRecursion
{
public:
    LegendreIntegralRecursion() = default;
    explicit LegendreIntegralRecursion(std::size_t order);

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    void expand(std::size_t order);

    template <typename T>
    void generate(std::span<T> values, T x)
    {
        if (values.size() == 0) return;

        std::size_t lmax = values.size() - 1;
        expand(lmax);

        values[0] = x + 1.0;

        if (lmax == 0) return;
        values[1] = 0.5*(x - 1.0)*(x + 1.0);

        for (std::size_t l = 2; l <= lmax; ++l)
            values[l] = m_a[l]*x*values[l - 1] - m_b[l]*values[l - 2];
    }

private:
    std::vector<double> m_a;
    std::vector<double> m_b;
    std::size_t m_order{};
};

} // namespace zdm::zebra
