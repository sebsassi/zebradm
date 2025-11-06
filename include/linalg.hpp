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

/**
    @brief Concept defining a static matrix-like type.

    A matrix-like object 'm' of type `T` has an arithemtic value type, and
    implements a two-dimensional subscript operator `m[i,j]`, where `i` and `j`
    are indices whose type is `T::size_type`. The subscripting should return
    `T::reference`. In addition, for the object to be static, its shape must be
    known at compile time, such that there is an array-like `static constexpr`
    member `T::shape`, whose members have type `T::size_type`.
*/
template <typename T>
concept static_matrix_like = std::is_arithmetic_v<typename T::value_type>
    && (std::tuple_size_v<decltype(T::shape)> == 2)
    && std::same_as<std::remove_cvref_t<decltype(T::shape[0])>, typename T::size_type>
    && std::same_as<std::remove_cvref_t<decltype(T::shape[1])>, typename T::size_type>
    && requires (T& matrix, typename T::size_type i, T::size_type j)
    {
        typename T::transpose_type;
        { matrix[i, j] } -> std::same_as<typename T::reference>;
    };

/**
    @brief Concept defining a static square-matrix-like object.

    A static square matrix-like object is a static matrix-like object whose
    shape has the same value in both dimensions.
*/
template <typename T>
concept static_square_matrix_like = static_matrix_like<T> && (T::shape[0] == T::shape[1]);

/**
    @brief Concept defining a static vector-like type.

    A vector like object `v` of type `T` has an arithmetic value type and
    implements a subscript operator `v[i]`, which takes in an index of type
    `T::size_type` and returns `T::reference`. In addition, it is static if it
    has a specialization for `std::tuple_size`.
*/
template <typename T>
concept static_vector_like = std::is_arithmetic_v<typename T::value_type>
    && std::same_as<std::remove_const_t<decltype(std::tuple_size_v<T>)>, typename T::size_type>
    && requires (T vector, typename T::size_type i)
    {
        { vector[i] } -> std::same_as<typename T::reference>;
    };

namespace detail
{

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T minus_(
    const T& a, std::index_sequence<idx...>) noexcept
{
    return {{(a[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a + b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a - b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a*b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a/b[idx])...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b)...}};
}

template <static_vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T::value_type dot_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
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
template <static_vector_like T>
[[nodiscard]] constexpr T::value_type dot(const T& a, const T& b) noexcept
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
template <static_vector_like T>
    requires (std::tuple_size_v<T> == 2)
[[nodiscard]] constexpr T cross(const T& a, const T& b) noexcept
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
[[nodiscard]] inline T::value_type length(const T& a) noexcept
{
    // NOTE: This function should be constexpr when clang decides to support
    // constexpr math.
    return std::sqrt(dot(a, a));
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

/**
    @brief Multiply a square matrix by a vector.

    @tparam T Square matrix type.
    @tparam U Vector type.

    @param mat Matrix.
    @param vec Vector.

    @return Product vector.

    This function returns the product \f$\vec{b} = M\vec{a}\f$.
*/
template <static_square_matrix_like T, static_vector_like U>
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

/**
    @brief Multiply two square matrices.

    @tparam T Square matrix type.

    @param a
    @param b

    @return Product matrix.

    This function returns the product \f$M = M_1M_2\f$.
*/
template <static_square_matrix_like T>
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

/**
    @brief Take transpose of a matrix.

    @tparam T Matrix type with a `transpose_type`.

    @param matrix

    @return Transpose of the matrix.
*/
template <static_matrix_like T>
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

/**
    @brief Normalize a vector

    @tparam Vector type.

    @param a

    @return Normalized vector.

    Given a vector \f$\vec{a}\f$ this function computes the normalized vector
    \f$\hat{n} = \vec{a}/|\vec{a}|\f$.
*/
template <static_vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T normalize(const T& a) noexcept
{
    const typename T::value_type norm = length(a);
    if (norm == typename T::value_type{}) return T{};

    return mul((1.0/norm), a);
}

/**
    @brief Evaluatea quadratic form defined by a matrix with a vector argument.

    @tparam T Square matrix type.
    @tparam U Vector type.

    @param matrix
    @param vector

    @return Value of quadratic form.

    Given a matrix \f$M\f$ and a vector \f$\vec{v}\f$ this function evaluates
    the quadratic form \f$\vec{v}^\mathsf{T}M\vec{v}\f$.
*/
template <static_square_matrix_like T, static_vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr T::value_type
quadratic_form(T matrix, U vector) noexcept
{
    return dot(vector, matmul(matrix, vector));
}

} // namespace la

} // namespace zdm
