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
#include <concepts>
#include <type_traits>

namespace zdm
{

enum class Axis { x, y, z };

namespace la
{

template <typename T>
concept matrix_like = std::is_arithmetic_v<typename T::value_type>
    && std::same_as<std::remove_const_t<decltype(T::shape)>, std::array<typename T::size_type, 2>>
    && requires (T matrix, typename T::size_type i, T::size_type j)
    {
        typename T::transpose_type;
        { matrix[i, j] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
    };

template <typename T>
concept square_matrix_like = matrix_like<T> && (T::shape[0] == T::shape[1]);

template <typename T>
concept vector_like = std::is_arithmetic_v<typename T::value_type>
    && std::same_as<std::remove_const_t<decltype(std::tuple_size_v<T>)>, typename T::size_type>
    && requires (T vector, typename T::size_type i)
    {
        { vector[i] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
    };

namespace detail
{

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T minus_(
    const T& a, std::index_sequence<idx...>) noexcept
{
    return {{(a[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a + b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a - b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a*b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a/b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T::value_type dot_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

template <vector_like T>
[[nodiscard]] constexpr T::value_type dot(const T& a, const T& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
    requires (std::tuple_size_v<T> == 2)
[[nodiscard]] constexpr T cross(const T& a, const T& b) noexcept
{
    return a[0]*b[1] - a[1]*b[0];
}

template <vector_like T>
    requires (std::tuple_size_v<T> == 3)
[[nodiscard]] constexpr T cross(const T& a, const T& b) noexcept
{
    T res{};
    res[0] = a[1]*b[2] - a[2]*b[3];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
    return res;
}

template <vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T::value_type length(const T& a) noexcept
{
    // NOTE: This function should be constexpr when clang decides to support
    // constexpr math.
    return std::sqrt(dot(a, a));
}

template <vector_like T>
[[nodiscard]] constexpr T
minus(const T& a) noexcept
{
    return detail::minus_(a, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
constexpr T&
add_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b[i];
    return a;
}

template <vector_like T>
constexpr T&
sub_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b[i];
    return a;
}

template <vector_like T>
constexpr T&
mul_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b[i];
    return a;
}

template <vector_like T>
constexpr T&
div_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b[i];
    return a;
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const typename T::value_type& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const typename T::value_type& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const typename T::value_type& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const typename T::value_type& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const typename T::value_type& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const typename T::value_type& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const typename T::value_type& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const typename T::value_type& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
constexpr T& add_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b;
    return a;
}

template <vector_like T>
constexpr T& sub_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b;
    return a;
}

template <vector_like T>
constexpr T& mul_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b;
    return a;
}

template <vector_like T>
constexpr T& div_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b;
    return a;
}


template <matrix_like T, vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[0] != T::shape[1]) && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr std::array<typename U::value_type, T::shape[0]>
matmul(const T& mat, const U& vec) noexcept
{
    std::array<typename U::value_type, T::shape[0]> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

template <square_matrix_like T, vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr U
matmul(const T& mat, const U& vec) noexcept
{
    U res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

template <square_matrix_like T>
[[nodiscard]] constexpr T
matmul(const T& a, const T& b) noexcept
{
    constexpr auto dimension = T::shape[0];
    T res{};
    for (std::size_t i = 0; i < dimension; ++i)
    {
        for (std::size_t j = 0; j < dimension; ++j)
        {
            for (std::size_t k = 0; k < dimension; ++k)
                res[i, j] += a[i, k]*b[k, j];
        }
    }

    return res;
}

template <matrix_like T>
[[nodiscard]] constexpr T::transpose_type
transpose(const T& matrix) noexcept
{
    typename T::transpose_type res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[j, i] = matrix[i, j];
    }
    return res;
}

template <vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T normalize(const T& a) noexcept
{
    const typename T::value_type norm = length(a);
    if (norm == typename T::value_type{}) return T{};

    return mul((1.0/norm), a);
}

template <square_matrix_like T, vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr T::value_type
quadratic_form(T matrix, U vector) noexcept
{
    return dot(vector, matmul(matrix, vector));
}

} // namespace la

} // namespace zdm
