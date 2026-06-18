/*
Copyright (c) 2025 Sebastian Sassi

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

#include "transform_conventions.hpp"

namespace zdm::la
{

/**
    @brief A type representing an identity operator.
*/
struct Identity
{
    template <typename T>
    [[nodiscard]] static constexpr T operator()(T v) { return v; }

    [[nodiscard]] static constexpr Identity inverse() { return Identity{}; }
};

/**
    @brief Composition of two identity operators.
*/
template <Chaining chaining>
[[nodiscard]] constexpr Identity compose([[maybe_unused]] Identity id1, [[maybe_unused]] Identity id2)
{
    return {};
}

/**
    @brief Composition of an identity operator with any other operator.
*/
template <Chaining chaining, typename T>
[[nodiscard]] constexpr T compose([[maybe_unused]] Identity id, T op)
{
    return op;
}

/**
    @brief Composition of any other operator with an identity operator.
*/
template <Chaining chaining, typename T>
[[nodiscard]] constexpr T compose(T op, [[maybe_unused]] Identity id)
{
    return op;
}

} // namespace zdm::la
