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

#include <format>
#include <source_location>
#include <span>
#include <vector>
#include <zest/md_array.hpp>

namespace zdm::util
{

[[nodiscard]] std::string format_error(
    std::string_view error_type, const std::source_location& location, std::string_view message);

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

template <typename ElementType, std::size_t N>
class SwapChain
{
public:
    SwapChain() = default;
    SwapChain(std::size_t size): m_buffer{size}
    {
        for (std::size_t i = 0; i < N; ++i)
            m_chain[i] = i;
    }

    void resize(std::size_t size) { m_buffer.reshape(size); }

    [[nodiscard]] std::size_t buffer_size() const noexcept { return m_buffer.extent(1); }

    template <std::size_t index>
        requires (index < N)
    [[nodiscard]] std::span<double> previous() noexcept { return m_buffer[m_chain[index]].flatten(); }

    template <std::size_t index>
        requires (index < N)
    [[nodiscard]] std::span<const ElementType> previous() const noexcept { return m_buffer[m_chain[index]].flatten(); }

    [[nodiscard]] std::span<ElementType> current() noexcept { return m_buffer[m_chain[0]].flatten(); }
    [[nodiscard]] std::span<const ElementType> current() const noexcept { return m_buffer[m_chain[0]].flatten(); }

    [[nodiscard]] std::span<ElementType> next() noexcept { return m_buffer[m_chain.back()].flatten(); }
    [[nodiscard]] std::span<const ElementType> next() const noexcept { return m_buffer[m_chain.back()].flatten(); }

    void advance()
    {
        const std::size_t back = m_chain.back();
        for (std::size_t i = N; i > 0; --i)
            m_chain[i] = m_chain[i - 1];

        m_chain[0] = back;
    }

private:
    zest::MDArray<ElementType, N, std::dynamic_extent> m_buffer;
    std::array<std::size_t, N> m_chain;
};

} // namespace zdm::util
