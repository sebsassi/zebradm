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

#include <concepts>
#include <expected>
#include <ranges>

#include "mp-units/concepts.h"

namespace zdm
{

template <typename T>
concept unary_arithmetic = requires (T x)
{
    { +x } -> std::same_as<T>;
    { -x } -> std::same_as<T>;
};

template <typename T>
concept basic_arithmetic = requires (T x, T y)
{
    { x + y } -> std::same_as<T>;
    { x - y } -> std::same_as<T>;
    { x*y } -> std::same_as<T>;
    { x/y } -> std::same_as<T>;
};

template <typename T>
concept assignable_basic_arithmetic = requires (T x, T y)
{
    { x += y } -> std::same_as<T&>;
    { x -= y } -> std::same_as<T&>;
    { x *= y } -> std::same_as<T&>;
    { x /= y } -> std::same_as<T&>;
};

template <typename T>
concept modular_arithmetic = requires (T x, T y)
{
    { x % y } -> std::same_as<T>;
};

template <typename T>
concept assignable_modular_arithmetic = requires (T x, T y)
{
    { x %= y } -> std::same_as<T&>;
};

template <typename T>
concept bitwise_arithmetic = requires (T x, T y)
{
    { x & y } -> std::same_as<T>;
    { x | y } -> std::same_as<T>;
    { x ^ y } -> std::same_as<T>;
    { x << y } -> std::same_as<T>;
    { x >> y } -> std::same_as<T>;
};

template <typename T>
concept assignable_bitwise_arithmetic = requires (T x, T y)
{
    { x &= y } -> std::same_as<T&>;
    { x |= y } -> std::same_as<T&>;
    { x ^= y } -> std::same_as<T&>;
    { x <<= y } -> std::same_as<T&>;
    { x >>= y } -> std::same_as<T&>;
};

template <typename T>
concept conventional_arithmetic = requires (T x, T y)
{
    { +x } -> std::same_as<std::remove_cvref_t<T>>;
    { -x } -> std::same_as<std::remove_cvref_t<T>>;

    { x + y } -> std::same_as<std::remove_cvref_t<T>>;
    { x - y } -> std::same_as<std::remove_cvref_t<T>>;
    { x*y } -> std::same_as<std::remove_cvref_t<T>>;
    { x/y } -> std::same_as<std::remove_cvref_t<T>>;

    { x += y } -> std::same_as<std::remove_cvref_t<T>&>;
    { x -= y } -> std::same_as<std::remove_cvref_t<T>&>;
    { x *= y } -> std::same_as<std::remove_cvref_t<T>&>;
    { x /= y } -> std::same_as<std::remove_cvref_t<T>&>;
};

namespace la
{

template <typename T>
inline constexpr std::size_t tensor_rank = []{ throw "no specialization for tensor_rank"; return 0; }();

template <typename T, std::size_t I>
inline constexpr std::size_t tensor_extent = []{ throw "no specialization for tensor_extent"; return 0; }();

/**
    @brief Concept defining a static matrix-like type.
*/
template <typename T>
concept static_matrix_like
    = requires (T& matrix, typename T::size_type i, T::size_type j) { matrix[i, j]; };

/**
    @brief Concept defining a static square-matrix-like object.

    A static square matrix-like object is a static matrix-like object whose
    shape has the same value in both dimensions.
*/
template <typename T>
concept static_square_matrix_like = static_matrix_like<T> && (tensor_extent<T, 0> == tensor_extent<T, 1>);

template <static_matrix_like T>
inline constexpr std::size_t tensor_rank<T> = 2;

template <static_matrix_like T, std::size_t I>
    requires (I < tensor_rank<T>) && std::unsigned_integral<decltype(T::template extent<I>)>
inline constexpr std::size_t tensor_extent<T, I> = T::template extent<I>;

/**
    @brief Concept defining a static vector-like type.
*/
template <typename T>
concept static_vector_like = requires (T vector, typename T::size_type i) { vector[i]; };

template <static_vector_like T>
inline constexpr std::size_t tensor_rank<T> = 1;

template <static_vector_like T, std::size_t I>
    requires (I < tensor_rank<T>)
inline constexpr std::size_t tensor_extent<T, I> = std::tuple_size_v<T>;

} // namespace la

namespace detail
{

template <typename T>
struct remove_unit_helper
{
    using type = T;
};

template <mp_units::Quantity T>
struct remove_unit_helper<T>
{
    using type = T::rep;
};

} // namespace detail

template <typename T>
using remove_unit = detail::remove_unit_helper<T>::type;

template <typename T, typename ErrorType>
concept ExpectedWith = std::same_as<T, std::expected<typename T::value_type, ErrorType>>;

template <typename T, template <typename> typename ContainerTemplate>
concept Container
    = std::ranges::range<ContainerTemplate<typename T::value_type>>
        && std::same_as<T, ContainerTemplate<typename T::value_type>>;

template <typename T, auto quantity_spec, typename ErrorType>
concept ExpectedQuantityOf
    = ExpectedWith<T, ErrorType>
        && mp_units::QuantityOf<typename T::value_type, quantity_spec>;

template <typename T, auto quantity_spec, template <typename> typename ContainerTemplate>
concept QuantityContainerOf
    = mp_units::QuantityOf<typename T::value_type, quantity_spec>
        && Container<T, ContainerTemplate>;

template <typename T, auto quantity_spec, template <typename> typename ContainerTemplate, typename ErrorType>
concept ExpectedQuantityContainerOf
    = ExpectedWith<T, ErrorType>
        && QuantityContainerOf<typename T::value_type, quantity_spec, ContainerTemplate>;

} // namespace zdm
