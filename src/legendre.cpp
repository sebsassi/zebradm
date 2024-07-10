#include "legendre.hpp"

#include <stdexcept>

LegendreArrayRecursion::LegendreArrayRecursion(std::size_t size):
    m_buffers{std::vector<double>(size), std::vector<double>(size), std::vector<double>(size)}, m_x(size), m_size(size) {}

LegendreArrayRecursion::LegendreArrayRecursion(std::span<const double> x):
    m_buffers{std::vector<double>(x.size()), std::vector<double>(x.size()), std::vector<double>(x.size())}, m_x(x.size()), m_size(x.size())
{
    init(x);
}

void LegendreArrayRecursion::resize(std::size_t size)
{
    if (size != m_size)
    {
        for (auto& buffer : m_buffers)
            buffer.resize(size);
        m_x.resize(size);
        m_size = size;
    }
}

void LegendreArrayRecursion::init(std::span<const double> x)
{
    resize(x.size());

    std::ranges::fill(m_buffers[0], 1.0);
    std::ranges::copy(x, m_x.begin());
    std::ranges::copy(m_x, m_buffers[1].begin());
    reset();
}

std::span<const double> LegendreArrayRecursion::next()
{
    double* temp = m_second_prev;
    m_second_prev = m_prev;
    m_prev = m_next;
    m_next = temp;

    if (m_l > 1)
    {
        const double inv_l = 1.0/double(m_l);
        const double a = double(2*m_l - 1)*inv_l;
        const double b = double(m_l - 1)*inv_l;
        for (std::size_t i = 0; i < m_size; ++i)
            m_next[i] = a*m_x[i]*m_prev[i] - b*m_second_prev[i];
    }
    ++m_l;

    return std::span<double>(m_next, m_size);
}


void LegendreArrayRecursion::reset()
{
    m_second_prev = m_buffers[0].data();
    m_prev = m_buffers[1].data();
    m_next = m_buffers[2].data();
    m_l = 0;
}

LegendreIntegralRecursion::LegendreIntegralRecursion(std::size_t order):
    m_a(order), m_b(order), m_order(order)
{
    for (std::size_t l = 2; l < order; ++l)
    {
        const double dl = double(l);
        m_a[l] = (2.0*dl - 1.0)/(dl + 1.0);
        m_b[l] = (dl - 2.0)/(dl + 1.0);
    }
}

void LegendreIntegralRecursion::expand(std::size_t order)
{
    if (order <= m_order) return;
    m_a.resize(order);
    m_b.resize(order);

    for (std::size_t l = m_order; l < order; ++l)
    {
        const double dl = double(l);
        m_a[l] = (2.0*dl - 1.0)/(dl + 1.0);
        m_b[l] = (dl - 2.0)/(dl + 1.0);
    }

    m_order = order;
}

void LegendreIntegralRecursion::legendre_integral(
    std::span<double> pl, double x)
{
    if (pl.size() == 0) return;

    std::size_t lmax = pl.size() - 1;
    expand(lmax);

    pl[0] = x + 1.0;

    if (lmax == 0) return;
    pl[1] = 0.5*(x - 1.0)*(x + 1.0);

    for (std::size_t l = 2; l <= lmax; ++l)
        pl[l] = m_a[l]*x*pl[l - 1] - m_b[l]*pl[l - 2];
}