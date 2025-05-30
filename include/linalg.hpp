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

#include <array>
#include <cmath>

namespace zdm
{
    
template <typename T, std::size_t N, std::size_t M>
using Matrix = std::array<std::array<T, M>, N>;

namespace detail
{

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a + b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a - b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a*b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a/b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr T dot_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

template <typename T, std::size_t N>
[[nodiscard]] constexpr T dot(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] inline T length(const std::array<T, N>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
add_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
sub_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
mul_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
div_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b[i];
    return a;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& add_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& sub_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& mul_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& div_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b;
    return a;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr std::array<T, N>
matmul(const Matrix<T, N, M>& mat, const std::array<T, M>& vec) noexcept
{
    std::array<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = dot(mat[i], vec);
    
    return res;
}

template <typename T, std::size_t N, std::size_t M, std::size_t L>
[[nodiscard]] constexpr Matrix<T, N, L>
matmul(const Matrix<T, N, M>& a, const Matrix<T, M, L>& b) noexcept
{
    Matrix<T, N, L> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < L; ++j)
        {
            std::array<T, M> bj;
            for (std::size_t k = 0; k < M; ++k)
                bj[k] = b[k][j];
            res[i][j] = dot(a[i], bj);
        }
    }
    
    return res;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr Matrix<T, M, N>
transpose(const Matrix<T, N, M>& matrix)
{
    Matrix<T, M, N> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < M; ++j)
            res[j][i] = matrix[i][j];
    }
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N> normalize(const std::array<T, N>& a) noexcept
{
    const T norm = length(a);
    if (norm == T{}) return std::array<T, N>{};

    return mul((1.0/norm), a);
}

} // namespace zdm
