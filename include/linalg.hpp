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

namespace zdm
{

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

template <typename FieldType, std::size_t N, std::size_t M, std::size_t L>
[[nodiscard]] constexpr Matrix<FieldType, N, L>
matmul(const Matrix<FieldType, N, M>& a, const Matrix<FieldType, M, L>& b) noexcept
{
    Matrix<FieldType, N, L> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < L; ++j)
        {
            Vector<FieldType, M> bj;
            for (std::size_t k = 0; k < M; ++k)
                bj[k] = b[k][j];
            res[i][j] = dot(a[i], bj);
        }
    }
    
    return res;
}

template <typename FieldType, std::size_t N>
[[nodiscard]] constexpr Vector<FieldType, N> normalize(const Vector<FieldType, N>& a) noexcept
{
    const FieldType norm = length(a);
    if (norm == FieldType{}) return Vector<FieldType, N>{};

    return (1.0/norm)*a;
}

} // namespace zdm