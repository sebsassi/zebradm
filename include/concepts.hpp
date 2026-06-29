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
concept real_arithmetic
    = unary_arithmetic<T> && basic_arithmetic<T> && assignable_basic_arithmetic<T>;

} // namespace zdm
