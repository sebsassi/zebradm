#pragma once

#include <span>
#include <vector>

namespace zdm::util
{

/**
    @brief Create an evenly spaced interval of real numbers.

    @tparam T Type of elements of the interval.

    @param interval Destination buffer for interval.
    @param start Starting value of the interval.
    @param end End value of the interval.
*/
template <std::floating_point T>
constexpr void
linspace(std::span<T>& interval, T start, T stop) noexcept
{
    if (interval.size() == 0) return;
    if (interval.size() == 1)
    {
        interval.front() = start;
        return;
    }

    const T step = (stop - start)/T(interval.size() - 1);
    for (std::size_t i = 0; i < interval.size(); ++i)
        interval[i] = start + T(i)*step;

    interval.back() = stop;
}

/**
    @brief Create an evenly spaced interval of real numbers.

    @tparam T Type of elements of the interval.

    @param interval Destination buffer for interval.
    @param start Starting value of the interval.
    @param end End value of the interval.
*/
template <std::floating_point T>
[[nodiscard]] std::vector<T>
linspace(T start, T stop, std::size_t count)
{
    std::vector<T> res(count);
    linspace(std::span<T>(res), start, stop);
    return res;
}

} // namespace zdm::util
