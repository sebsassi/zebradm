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

#include <cmath>
#include <concepts>
#include <type_traits>

#include "concepts.hpp"

namespace zdm
{

enum class Axis { x, y, z };

namespace la
{

namespace detail
{

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T minus_(
    const T& a, std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(-a[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a + b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx] + b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a - b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx] - b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a*b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx]*b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a/b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return {{(a[idx]/b)...}};
}

template <static_vector_like T, static_vector_like U, std::size_t... idx>
    requires (std::tuple_size_v<T> == std::tuple_size_v<U>)
[[nodiscard]] constexpr auto dot_(
    const T& a, const U& b, 
    std::index_sequence<idx...> /*unused*/) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

/**
    @brief Vector dot product.

    @tparam T Vector type.

    @param a
    @param b

    This function evaluates the dot product \f$\vec{a}\cdot \vec{b}\f$ of two
    vectors \f$a\f$ and \f$b\f$.
*/
template <static_vector_like T, static_vector_like U>
    requires (std::tuple_size_v<T> == std::tuple_size_v<U>)
[[nodiscard]] constexpr auto dot(const T& a, const U& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Vector cross product.


    @tparam T Vector type.

    @param a
    @param b

    For two dimensional vectors \f$\vec{a}\f$ and \f$\vec{b}\f$ this function
    returns the value \f$a_1b_2 - a_2b_1\f$. For three dimensional vectors it
    returns the vector
    \f[
        \vec{a}\times\vec[b] = (a_2b_3 - a_3b2, a_3b_1 - a-1b_3, a_1b_2 - a_2b_1).
    \f]
*/
template <static_vector_like T, static_vector_like U>
    requires (std::tuple_size_v<T> == 2 && std::tuple_size_v<U> == 2)
[[nodiscard]] constexpr T cross(const T& a, const U& b) noexcept
{
    return a[0]*b[1] - a[1]*b[0];
}

/**
    @brief Vector cross product.


    @tparam T Vector type.

    @param a
    @param b

    For two dimensional vectors \f$\vec{a}\f$ and \f$\vec{b}\f$ this function
    returns the value \f$a_1b_2 - a_2b_1\f$. For three dimensional vectors it
    returns the vector
    \f[
        \vec{a}\times\vec[b] = (a_2b_3 - a_3b2, a_3b_1 - a-1b_3, a_1b_2 - a_2b_1).
    \f]
*/
template <static_vector_like T>
    requires (std::tuple_size_v<T> == 3)
[[nodiscard]] constexpr T cross(const T& a, const T& b) noexcept
{
    T res{};
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
    return res;
}

/**
    @brief Length of a vector.

    @tparam T Floating point vector type.

    @param a

    Calculates the length of the vector \f$\vec{a}\f$,
    \f[
        |\vec{a}| = \sqrt{\vec{a}\cdot\vec{a}}.
    \f]
    This operation has been restricted to floating point vectors, because
    if you are trying to take the Euclidean length of an integer vector, you're
    probably doing something wrong.
*/
template <static_vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T::value_type norm(const T& a) noexcept
{
    // NOTE: This function should be constexpr when clang decides to support
    // constexpr math.
    return std::sqrt(dot(a, a));
}

/**
    @brief Negation of a vector.

    @tparam T Vector type.

    @param a

    @return `+a`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
plus(const T& a) noexcept
{
    return a;
}

/**
    @brief Negation of a vector.

    @tparam T Vector type.

    @param a

    @return `-a`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
minus(const T& a) noexcept
{
    return detail::minus_(a, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Vector addition.

    @tparam T Vector type.

    @param a
    @param b

    @return `a + b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Vector subtraction.

    @tparam T Vector type.

    @param a
    @param b

    @return `a - b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Elementwise vector multiplication.

    @tparam T Vector type.

    @param a
    @param b

    @return `a*b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Elementwise vector division.

    @tparam T Vector type.

    @param a
    @param b

    @return `a/b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Add vector to another in place.

    @tparam T Vector type.

    @param a
    @param b

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T&
add_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b[i];
    return a;
}

/**
    @brief Subtract vector from another in place.

    @tparam T Vector type.

    @param a
    @param b

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T&
sub_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b[i];
    return a;
}

/**
    @brief Multiply vector elementwise with another in place.

    @tparam T Vector type.

    @param a
    @param b

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T&
mul_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b[i];
    return a;
}

/**
    @brief Divide vector elementwise by another in place.

    @tparam T Vector type.

    @param a
    @param b

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T&
div_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b[i];
    return a;
}

/**
    @brief Add scalar and a vector.

    @tparam T Vector type.

    @param a Scalar.
    @param b Vector.

    @return `a + b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
add(const typename T::value_type& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Subtract a scalar and a vector.

    @tparam T Vector type.

    @param a Scalar.
    @param b Vector.

    @return `a - b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
sub(const typename T::value_type& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Multiply a scalar and a vector.

    @tparam T Vector type.

    @param a Scalar.
    @param b Vector.

    @return `a*b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
mul(const typename T::value_type& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Divide a scalar and a vector, element-wise.

    @tparam T Vector type.

    @param a Scalar.
    @param b Vector.

    @return `a/b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
div(const typename T::value_type& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Add a vector and a scalar.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return `a + b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const typename T::value_type& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Subtract a vector and a scalar.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return `a - b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const typename T::value_type& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Multiply a vector and a scalar.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return `a*b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const typename T::value_type& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Divide a vector by a scalar.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return `a*b`.
*/
template <static_vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const typename T::value_type& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Add a scalar to a vector in place.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T& add_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b;
    return a;
}

/**
    @brief Subtract a scalar from a vector in place.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T& sub_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b;
    return a;
}

/**
    @brief Multiply a scalar to a vector in place.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T& mul_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b;
    return a;
}

/**
    @brief Divide a vector by a scalar in place.

    @tparam T Vector type.

    @param a Vector.
    @param b Scalar.

    @return reference to `a`.
*/
template <static_vector_like T>
constexpr T& div_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b;
    return a;
}

namespace detail
{

template <static_square_matrix_like T, static_vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (tensor_extent<T, 1> == tensor_extent<U, 0>)
[[nodiscard]] constexpr U
matmul_(const T& mat, const U& vec) noexcept
{
    U res{};
    for (std::size_t i = 0; i < tensor_extent<T, 0>; ++i)
    {
        for (std::size_t j = 0; j < tensor_extent<T, 1>; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

} // namespace detail

/**
    @brief Multiply a square matrix by a vector.

    @param mat Matrix.
    @param vec Vector.

    @return Product vector.

    This function returns the product \f$\vec{b} = M\vec{a}\f$.
*/
[[nodiscard]] constexpr static_vector_like auto
matmul(const static_matrix_like auto& mat, const static_vector_like auto& vec) noexcept
    requires requires { { mat*vec } -> static_vector_like; }
{
    return mat*vec;
}

/**
    @brief Multiply two square matrices.

    @param a
    @param b

    @return Product matrix.

    This function returns the product \f$M = M_1M_2\f$.
*/
[[nodiscard]] constexpr static_matrix_like auto
matmul(const static_matrix_like auto& a, const static_matrix_like auto& b) noexcept
    requires requires { { a*b } -> static_matrix_like; }
{
    return a*b;
}

/**
    @brief Take transpose of a matrix.

    @tparam T Matrix type with a `transpose_type`.

    @param matrix

    @return Transpose of the matrix.
*/
[[nodiscard]] constexpr static_matrix_like auto
transpose(const static_matrix_like auto& matrix) noexcept
    requires requires { { matrix.transpose() } -> static_matrix_like; }
{
    return matrix.transpose();
}

/**
    @brief Normalize a vector

    @param a

    @return Normalized vector.

    Given a vector \f$\vec{a}\f$ this function computes the normalized vector
    \f$\hat{n} = \vec{a}/|\vec{a}|\f$.
*/
[[nodiscard]] inline auto normalize(const static_vector_like auto& a) noexcept
    requires requires { norm(a); }
{
    const auto a_norm = norm(a);
    if (a_norm == decltype(a_norm){}) return decltype(a){};

    return (1.0/a_norm)*a;
}

/**
    @brief Evaluatea quadratic form defined by a matrix with a vector argument.

    @param matrix
    @param vector

    @return Value of quadratic form.

    Given a matrix \f$M\f$ and a vector \f$\vec{v}\f$ this function evaluates
    the quadratic form \f$\vec{v}^\mathsf{T}M\vec{v}\f$.
*/
[[nodiscard]] constexpr auto
quadratic_form(
    const static_square_matrix_like auto& matrix,
    const static_vector_like auto& vector) noexcept
{
    return dot(vector, matmul(matrix, vector));
}

} // namespace la

} // namespace zdm
