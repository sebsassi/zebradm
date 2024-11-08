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

namespace cubage
{

template <typename T>
concept arithmetic = std::is_arithmetic<T>::value;

template <std::floating_point T, std::size_t Count, std::size_t Dim>
constexpr auto& orthonormalize(std::array<std::array<T, Dim>, Count>& basis)
{
    for (std::size_t i = 0; i < Count - 1; ++i)
    {
        normalize(basis[i]);
        for (std::size_t j = i + 1; j < Count; ++j)
            basis[j] -= project_along_unitv(basis[j], basis[i]);
    }
}

template <std::floating_point T, std::size_t N>
constexpr std::array<T, N> project_along_unitv(
    const std::array<T, N>& a, const std::array<T, N>& uvec)
{
    const T proj = dot(a, uvec);
    std::array<T, N> res = uvec;
    for (auto& element : res)
        element *= proj;
    return res;
}

template <std::floating_point T, std::size_t N>
constexpr std::array<T, N>& normalize(std::array<T, N>& a)
{
    T inv_length = 1.0/length(a);
    for (auto& element : a)
        element *= inv_length;
    return a;
}

template <arithmetic T, std::size_t N>
constexpr T dot(const std::array<T, N>& a, const std::array<T, N>& b)
{
    T sum = 0;
    for (std::size_t i = 0; i < N; ++i)
        sum += a[i]*b[i];
    return sum;
}

template <arithmetic T, std::size_t N>
constexpr T triple_dot(
    const std::array<T, N>& a, const std::array<T, N>& b,
    const std::array<T, N>& c)
{
    T sum = 0;
    for (std::size_t i = 0; i < N; ++i)
        sum += a[i]*b[i]*c[i];
    return sum;
}

template <std::floating_point T, std::size_t N>
constexpr T length(const std::array<T, N>& a)
{
    T largest = *std::ranges::max_element(a);
    if (largest == 0.0) return 0.0;
    T inv_largest = 1.0/largest;
    T sqr_sum = 0.0;
    for (const auto& element : a)
    {
        T scaled = element*inv_largest;
        sqr_sum += scaled*scaled;
    }
    return largest*std::sqrt(sqr_sum);
}

}