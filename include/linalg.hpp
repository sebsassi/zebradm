#pragma once

#include <array>

template <typename FieldType, std::size_t N>
using Vector = std::array<FieldType, N>;

template <typename FieldType, std::size_t N, std::size_t M>
using Matrix = std::array<std::array<FieldType, M>, N>;

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr FieldType
dot(const Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    FieldType res{};
    for (std::size_t i = 0; i < N; ++i)
        res += a[i]*b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] inline FieldType length(const Vector<FieldType, N>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator+(const Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator-(const Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator*(const Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator/(const Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b[i];
    return res;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>&
operator+=(Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b[i];
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>&
operator-=(Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b[i];
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>&
operator*=(Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b[i];
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>&
operator/=(Vector<FieldType, N>& a, const Vector<FieldType, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b[i];
    return a;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator+(const FieldType& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a + b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator-(const FieldType& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a - b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator*(const FieldType& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a*b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator/(const FieldType& a, const Vector<FieldType, N>& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a/b[i];
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator+(const Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b;
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator-(const Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b;
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator*(const Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b;
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N>
operator/(const Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b;
    return res;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>& operator+=(Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b;
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>& operator-=(Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b;
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>& operator*=(Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b;
    return a;
}

template <typename FieldType, std::size_t N>
constexpr Vector<FieldType, N>& operator/=(Vector<FieldType, N>& a, const FieldType& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b;
    return a;
}

template <typename FieldType, std::size_t N, std::size_t M>
[[nodiscard]] constexpr Vector<FieldType, N>
matmul(const Matrix<FieldType, N, M>& mat, const Vector<FieldType, M>& vec) noexcept
{
    Vector<FieldType, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = dot(mat[i], vec);
    
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N> normalize(const Vector<FieldType, N>& a) noexcept
{
    const FieldType norm = length(a);
    if (norm == FieldType{}) return Vector<FieldType, N>{};

    return (1.0/norm)*a;
}