/*
Copyright (c) 2024 Sebastian Sassi

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
#include <ranges>
#include <span>
#include <vector>

#include <zest/md_span.hpp>

namespace zdm::zebra
{

void legendre_recursion_vec(
    zest::MDSpan<double, 2> legendre, std::span<double> x);

void legendre_recursion(std::span<double> legendre, double x);


class LegendreArrayRecursion
{
public:
    LegendreArrayRecursion() = default;
    explicit LegendreArrayRecursion(std::size_t size);
    explicit LegendreArrayRecursion(std::span<const double> x);

    [[nodiscard]] std::size_t size() const noexcept { return m_size; }

    void resize(std::size_t size);
    void init(std::span<const double> x);

    template <typename Func>
        requires requires (Func f, std::span<double> x) { f(x); }
    void init(Func&& f) noexcept
    {
        std::ranges::fill(m_buffers[0], 1.0);
        f(m_x);
        std::ranges::copy(m_x, m_buffers[1].begin());
        reset();
    }

    [[nodiscard]] std::span<const double>
    second_prev() const noexcept { return std::span(m_second_prev, m_size); }

    [[nodiscard]] std::span<const double>
    prev() const noexcept { return std::span(m_prev, m_size); }

    [[nodiscard]] std::span<const double>
    current() const noexcept { return std::span(m_current, m_size); }

    void iterate() noexcept;
    void iterate(std::size_t n) noexcept;
    std::span<const double> next() noexcept;

private:
    void reset() noexcept;

    std::array<std::vector<double>, 3> m_buffers;
    std::vector<double> m_x;
    double* m_current;
    double* m_prev;
    double* m_second_prev;
    std::size_t m_size;
    std::size_t m_l = 0;
};

class LegendreIntegralRecursion
{
public:
    LegendreIntegralRecursion() = default;
    explicit LegendreIntegralRecursion(std::size_t order);

    void expand(std::size_t order);
    void legendre_integral(std::span<double> pl, double x);

private:
    std::vector<double> m_a;
    std::vector<double> m_b;
    std::size_t m_order;
};

} // namespace zdm::zebra
