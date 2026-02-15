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

#include "concepts.hpp"
#include <source_location>
#include <span>
#include <vector>
#include <zest/md_array.hpp>

#if defined(__GNUC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#endif

namespace zdm::util
{

/**
    @brief Check whether two spans overlap
*/
template <typename T, typename S>
    requires std::same_as<std::remove_cv_t<T>, std::remove_cv_t<S>>
[[nodiscard]] constexpr bool
have_overlap(std::span<T> a, std::span<S> b) noexcept
{
    const T* a_begin = a.data();
    const T* a_end = a.data() + a.size();
    const T* b_begin = b.data();
    const T* b_end = b.data() + b.size();
    return std::max(a_begin, b_begin) < std::min(a_end, b_end);
}

// multiply `b` to `a`: `a *= b`
template <arithmetic T>
constexpr void mul(std::span<T> a, std::span<const T> b) noexcept
{
    assert(!have_overlap(a, b));
    const std::size_t size = std::min(a.size(), b.size());
    [](T* RESTRICT a, const T* b, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] *= b[i];
    }(a.data(), b.data(), size);
}

template <arithmetic T>
constexpr void mul(std::span<T> a, std::span<T> b) noexcept
{
    mul(a, std::span<const T>(b));
}

// multiply `b` to `a`: `a *= b`
template <arithmetic T>
constexpr void mul(std::span<T> a, T b) noexcept
{
    const std::size_t size = a.size();
    [](T* RESTRICT a, T b, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] *= b;
    }(a.data(), b, size);
}

// multiply `b` to `a`: `a *= b`
template <arithmetic T>
constexpr void mul(std::span<T> a, std::span<const T> b, std::span<const T> c) noexcept
{
    assert(!have_overlap(a, b));
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(std::min(a.size(), b.size()), c.size());
    [](T* RESTRICT a, const T* b, const T* c, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] = b[i]*c[i];
    }(a.data(), b.data(), c.data(), size);
}

template <arithmetic T>
constexpr void mul(std::span<T> a, std::span<T> b, std::span<T> c) noexcept
{
    mul(a, std::span<const T>(b), std::span<const T>(c));
}

// multiply `b` to `a`: `a *= b`
template <arithmetic T>
constexpr void mul(std::span<T> a, T b, std::span<const T> c) noexcept
{
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(a.size(), c.size());
    [](T* RESTRICT a, T b, const T* c, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] = b*c[i];
    }(a.data(), b, c.data(), size);
}

template <arithmetic T>
constexpr void mul(std::span<T> a, T b, std::span<T> c) noexcept
{
    mul(a, b, std::span<const T>(c));
}

// multiply `c` and `b` and add to `a`: `a += b*c`
template <arithmetic T>
constexpr void fmadd(std::span<T> a, std::span<const T> b, std::span<const T> c) noexcept
{
    assert(!have_overlap(a, b));
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(std::min(a.size(), b.size()), c.size());
    [](T* RESTRICT a, const T* b, const T* c, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] += b[i]*c[i];
    }(a.data(), b.data(), c.data(), size);
}

template <arithmetic T>
constexpr void fmadd(std::span<T> a, std::span<T> b, std::span<T> c) noexcept
{
    fmadd(a, std::span<const T>(b), std::span<const T>(c));
}

// multiply `c` and `b` and add to `a`: `a += b*c`
template <arithmetic T>
constexpr void fmadd(std::span<T> a, T b, std::span<const T> c) noexcept
{
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(a.size(), c.size());
    [](T* RESTRICT a, T b, const T* c, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] += b*c[i];
    }(a.data(), b, c.data(), size);
}

template <arithmetic T>
constexpr void fmadd(std::span<T> a, T b, std::span<T> c) noexcept
{
    fmadd(a, b, std::span<const T>(c));
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
template <arithmetic T>
constexpr void fmadd(std::span<T> a, T b, T c, std::span<const T> d) noexcept
{
    assert(!have_overlap(a, d));
    const std::size_t size = std::min(a.size(), d.size());
    [](T* RESTRICT a, T b, T c, const T* d, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] = b + c*d[i];
    }(a.data(), b, c, d.data(), size);
}

template <arithmetic T>
constexpr void fmadd(std::span<T> a, T b, T c, std::span<T> d) noexcept
{
    fmadd(a, b, c, std::span<const T>(d));
}

// multiply `d` and `c`, add `b`, and save to `a`: `a = b + c*d`
template <arithmetic T>
constexpr void linear_combination(std::span<T> a, T b, std::span<const T> c, T d, std::span<const T> e) noexcept
{
    assert(!have_overlap(a, c));
    assert(!have_overlap(a, c));
    const std::size_t size = std::min(std::min(a.size(), c.size()), e.size());
    [](T* RESTRICT a, T b, const T* c, T d, const T* e, std::size_t size) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            a[i] = b*c[i] + d*e[i];
    }(a.data(), b, c.data(), d, e.data(), size);
}

template <arithmetic T>
[[nodiscard]] T
inner_product(std::span<const T> a, std::span<const T> b) noexcept
{
    const std::size_t size = std::min(a.size(), b.size());
    if (size == 0) return T{};

    std::array<T, 16> partial_res{};

    auto i = std::ptrdiff_t(size - 1);

    if (size > 16)
    {
        for (; i > 14; i -= 16)
        {
            partial_res[0] += a[i - 0]*b[i - 0];
            partial_res[1] += a[i - 1]*b[i - 1];
            partial_res[2] += a[i - 2]*b[i - 2];
            partial_res[3] += a[i - 3]*b[i - 3];
            partial_res[4] += a[i - 4]*b[i - 4];
            partial_res[5] += a[i - 5]*b[i - 5];
            partial_res[6] += a[i - 6]*b[i - 6];
            partial_res[7] += a[i - 7]*b[i - 7];
            partial_res[8] += a[i - 8]*b[i - 8];
            partial_res[9] += a[i - 9]*b[i - 9];
            partial_res[10] += a[i - 10]*b[i - 10];
            partial_res[11] += a[i - 11]*b[i - 11];
            partial_res[12] += a[i - 12]*b[i - 12];
            partial_res[13] += a[i - 13]*b[i - 13];
            partial_res[14] += a[i - 14]*b[i - 14];
            partial_res[15] += a[i - 15]*b[i - 15];
        }

        partial_res[0] += partial_res[8];
        partial_res[1] += partial_res[9];
        partial_res[2] += partial_res[10];
        partial_res[3] += partial_res[11];
        partial_res[4] += partial_res[12];
        partial_res[5] += partial_res[13];
        partial_res[6] += partial_res[14];
        partial_res[7] += partial_res[15];

        partial_res[0] += partial_res[4];
        partial_res[1] += partial_res[5];
        partial_res[2] += partial_res[6];
        partial_res[3] += partial_res[7];

        partial_res[0] += partial_res[2];
        partial_res[1] += partial_res[3];
    }

    for (; i > 0; i -= 2)
    {
        partial_res[0] += a[i - 0]*b[i - 0];
        partial_res[1] += a[i - 1]*b[i - 1];
    }

    if (i == 0)
        return partial_res[0] + partial_res[1] + a[0]*b[0];
    else
        return partial_res[0] + partial_res[1];
}

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
        for (std::size_t i = N - 1; i > 0; --i)
            m_chain[i] = m_chain[i - 1];

        m_chain[0] = back;
    }

private:
    zest::MDArray<ElementType, N, std::dynamic_extent> m_buffer;
    std::array<std::size_t, N> m_chain;
};

} // namespace zdm::util
