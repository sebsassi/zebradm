#pragma once

#include <cstddef>
#include <concepts>
#include <array>

template <typename FieldType>
concept arithmetic = std::is_arithmetic<FieldType>::value;

template <std::floating_point FieldType, std::size_t N>
constexpr std::array<FieldType, N>& normalize(std::array<FieldType, N>& a)
{
    FieldType inv_length = 1.0/length(a);
    for (auto& element : a)
        element *= inv_length;
    return a;
}

template <std::floating_point FieldType, std::size_t N>
constexpr std::array<FieldType, N> project_along_unitv(
    const std::array<FieldType, N>& a, const std::array<FieldType, N>& uvec)
{
    const FieldType proj = dot(a, uvec);
    std::array<FieldType, N> res = uvec;
    for (auto& element : res)
        element *= proj;
    return res;
}

template <std::floating_point FieldType, std::size_t Count, std::size_t Dim>
constexpr auto& orthonormalize(std::array<std::array<FieldType, Dim>, Count>& basis)
{
    for (std::size_t i = 0; i < Count - 1; ++i)
    {
        normalize(basis[i]);
        for (std::size_t j = i + 1; j < Count; ++j)
            basis[j] -= project_along_unitv(basis[j], basis[i]);
    }

    return basis;
}

template <arithmetic FieldType, std::size_t N>
constexpr FieldType dot(const std::array<FieldType, N>& a, const std::array<FieldType, N>& b)
{
    FieldType sum = 0;
    for (std::size_t i = 0; i < N; ++i)
        sum += a[i]*b[i];
    return sum;
}

template <arithmetic FieldType, std::size_t N>
constexpr FieldType triple_dot(
    const std::array<FieldType, N>& a, const std::array<FieldType, N>& b,
    const std::array<FieldType, N>& c)
{
    FieldType sum = 0;
    for (std::size_t i = 0; i < N; ++i)
        sum += a[i]*b[i]*c[i];
    return sum;
}

template <std::floating_point FieldType, std::size_t N>
constexpr FieldType length(const std::array<FieldType, N>& a)
{
    FieldType largest = *std::ranges::max_element(a);
    if (largest == 0.0) return 0.0;
    FieldType inv_largest = 1.0/largest;
    FieldType sqr_sum = 0.0;
    for (const auto& element : a)
    {
        FieldType scaled = element*inv_largest;
        sqr_sum += scaled*scaled;
    }
    return largest*std::sqrt(sqr_sum);
}