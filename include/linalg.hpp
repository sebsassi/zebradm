#pragma once

#include <array>

template <typename T, std::size_t N>
using Vector = std::array<T, N>;

template <typename T, std::size_t N, std::size_t M>
using Matrix = std::array<std::array<T, M>, N>;

template <typename T, std::size_t N>
[[nodiscard]] constexpr T
dot(const Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    T res{};
    for (std::size_t i = 0; i < N; ++i)
        res += a[i]*b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T length(const Vector<T, N>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator+(const Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator-(const Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator*(const Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator/(const Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b[i];
    return res;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>&
operator+=(Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>&
operator-=(Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>&
operator*=(Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>&
operator/=(Vector<T, N>& a, const Vector<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b[i];
    return a;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator+(const T& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a + b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator-(const T& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a - b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator*(const T& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a*b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator/(const T& a, const Vector<T, N>& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a/b[i];
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator+(const Vector<T, N>& a, const T& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b;
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator-(const Vector<T, N>& a, const T& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b;
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator*(const Vector<T, N>& a, const T& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b;
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N>
operator/(const Vector<T, N>& a, const T& b) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b;
    return res;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>& operator+=(Vector<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b;
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>& operator-=(Vector<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b;
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>& operator*=(Vector<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b;
    return a;
}

template <typename T, std::size_t N>
constexpr Vector<T, N>& operator/=(Vector<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b;
    return a;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr Vector<T, N>
matmul(const Matrix<T, N, M>& mat, const Vector<T, M>& vec) noexcept
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = dot(mat[i], vec);
    
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> normalize(const Vector<T, N>& a) noexcept
{
    const T norm = length(a);
    if (norm == T{}) return Vector<T, N>{};

    return (1.0/norm)*a;
}