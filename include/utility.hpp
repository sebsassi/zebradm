#pragma once

#include <span>
#include <vector>

namespace zdm::util
{

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

template <std::floating_point T>
[[nodiscard]] std::vector<T>
linspace(T start, T stop, std::size_t count)
{
    std::vector<T> res(count);
    linspace(std::span<T>(res), start, stop);
    return res;
}

} // namespace zdm::util
