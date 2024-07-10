#pragma once

#include <array>

template <typename T, std::size_t N>
using Vector = std::array<T, N>;

template <typename T, std::size_t N, std::size_t M>
using Matrix = std::array<std::array<T, M>, N>;

template <typename T, std::size_t N>
T dot(const Vector<T, N>& a, const Vector<T, N>& b)
{
    T res{};
    for (std::size_t i = 0; i < N; ++i)
        res += a[i]*b[i];
    return res;
}

template <typename T, std::size_t N>
T length(const Vector<T, N>& a)
{
    return std::sqrt(dot(a, a));
}

template <typename T, std::size_t N>
Vector<T, N> operator+(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator-(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator*(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator/(const Vector<T, N>& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N>& operator+=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b[i];
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator-=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b[i];
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator*=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b[i];
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator/=(Vector<T, N>& a, const Vector<T, N>& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b[i];
    return a;
}

template <typename T, std::size_t N>
Vector<T, N> operator+(const T& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a + b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator-(const T& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a - b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator*(const T& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a*b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator/(const T& a, const Vector<T, N>& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a/b[i];
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator+(const Vector<T, N>& a, const T& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] + b;
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator-(const Vector<T, N>& a, const T& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i] - b;
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator*(const Vector<T, N>& a, const T& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]*b;
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> operator/(const Vector<T, N>& a, const T& b)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = a[i]/b;
    return res;
}

template <typename T, std::size_t N>
Vector<T, N>& operator+=(Vector<T, N>& a, const T& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b;
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator-=(Vector<T, N>& a, const T& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b;
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator*=(Vector<T, N>& a, const T& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b;
    return a;
}

template <typename T, std::size_t N>
Vector<T, N>& operator/=(Vector<T, N>& a, const T& b)
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b;
    return a;
}

template <typename T, std::size_t N, std::size_t M>
Vector<T, N> matmul(const Matrix<T, N, M>& mat, const Vector<T, M>& vec)
{
    Vector<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
        res[i] = dot(mat[i], vec);
    
    return res;
}

template <typename T, std::size_t N>
Vector<T, N> normalize(const Vector<T, N>& a)
{
    const T norm = length(a);
    if (norm == T{}) return Vector<T, N>{};

    return (1.0/norm)*a;
}