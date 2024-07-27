#pragma once

#include <vector>
#include <span>
#include <algorithm>
#include <ranges>

#include "zest/md_span.hpp"

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

    std::span<const double> next() noexcept;
    void iterate(std::size_t n) noexcept;

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