#pragma once

#include <vector>
#include <span>
#include <algorithm>
#include <ranges>

void legendre_recursion_vec(
    zest::MDSpan<double, 2> legendre, std::span<double> x)
{
    const std::size_t order = legendre.extents()[0];
    const std::size_t size = legendre.extents()[1];

    if (order == 0) return;

    if (size != x.size())
        throw std::invalid_argument(
                "size of x is incompatible with size of leg");
    
    for (auto xi : x)
    {
        if (std::fabs(xi) > 1.0)
            throw std::invalid_argument("x must be between -1 and 1");
    }

    zest::MDSpan<double, 1> leg_0 = legendre[0];
    for (std::size_t i = 0; i < size; ++i)
        leg_0[i] = 1.0;
    
    if (order == 1) return;

    zest::MDSpan<double, 1> leg_1 = legendre[1];
    for (std::size_t i = 0; i < size; ++i)
        leg_1[i] = x[i];
    
    for (std::size_t n = 2; n < order; ++n)
    {
        zest::MDSpan<double, 1> leg_n = legendre[n];
        zest::MDSpan<double, 1> leg_nm1 = legendre[n - 1];
        zest::MDSpan<double, 1> leg_nm2 = legendre[n - 2];
        const double inv_n = 1.0/double(n);
        const double a = double(2*n - 1)*inv_n;
        const double b = double(n - 1)*inv_n;
        for (std::size_t i = 0; i < size; ++i)
            leg_n[i] = a*x[i]*leg_nm1[i] - b*leg_nm2[i];
    }
}

void legendre_recursion(std::span<double> legendre, double x)
{
    const std::size_t order = legendre.size();

    if (order == 0) return;
    
    if (std::fabs(x) > 1.0)
        throw std::invalid_argument("x must be between -1 and 1");

    legendre[0] = 1.0;
    if (order == 1) return;

    legendre[1] = x;
    
    for (std::size_t n = 2; n < order; ++n)
    {
        const double inv_n = 1.0/double(n);
        const double a = double(2*n - 1)*inv_n;
        const double b = double(n - 1)*inv_n;
        legendre[n] = a*x*legendre[n - 1] - b*legendre[n - 2];
    }
}


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
    void init(Func&& f)
    {
        std::ranges::fill(m_buffers[0], 1.0);
        f(m_x);
        std::ranges::copy(m_x, m_buffers[1].begin());
        reset();
    }

    std::span<const double> next();

private:
    void reset();

    std::array<std::vector<double>, 3> m_buffers;
    std::vector<double> m_x;
    double* m_next;
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