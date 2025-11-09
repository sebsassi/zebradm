/*
Copyright (c) 2025 Sebastian Sassi

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
