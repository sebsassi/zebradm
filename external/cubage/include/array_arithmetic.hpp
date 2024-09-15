#pragma once

#include <array>

template <typename FieldType>
concept ArithmeticAssignable = requires(FieldType a, FieldType b)
{
    a += b;
    a -= b;
    a *= b;
    a /= b;
};

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
constexpr FieldType& operator+=(
    FieldType& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size<FieldType>::value; ++i)
        a[i] += b[i];
    return a;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
[[nodiscard]] constexpr FieldType operator+(
    const FieldType& a, const FieldType& b) noexcept
{
    FieldType res = a;
    res += b;
    return res;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
constexpr FieldType& operator-=(
    FieldType& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size<FieldType>::value; ++i)
        a[i] -= b[i];
    return a;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
[[nodiscard]] constexpr FieldType operator-(
    const FieldType& a, const FieldType& b) noexcept
{
    FieldType res = a;
    res -= b;
    return res;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
constexpr FieldType& operator*=(FieldType& a, const typename FieldType::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size<FieldType>::value; ++i)
        a[i] *= b;
    return a;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
[[nodiscard]] constexpr FieldType operator*(
    const FieldType& a, const typename FieldType::value_type& b) noexcept
{
    FieldType res = a;
    res *= b;
    return res;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
[[nodiscard]] constexpr FieldType operator*(
    const typename FieldType::value_type& b, const FieldType& a) noexcept
{
    FieldType res = a;
    res *= b;
    return res;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
constexpr FieldType& operator*=(
    FieldType& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size<FieldType>::value; ++i)
        a[i] *= b[i];
    return a;
}

template <typename FieldType>
    requires std::same_as<FieldType, std::array<typename FieldType::value_type, std::tuple_size<FieldType>::value>> &&
    ArithmeticAssignable<typename FieldType::value_type>
[[nodiscard]] constexpr FieldType operator*(
    const FieldType& a, const FieldType& b) noexcept
{
    FieldType res = a;
    res *= b;
    return res;
}