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
#include "zest/zernike_expansion.hpp"
#include "zest/real_sh_expansion.hpp"
#include "zest/sh_glq_transformer.hpp"

#include "types.hpp"

namespace zdm
{
namespace zebra
{

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param in input Zernike expansion `f_{nlm}`
    @param out output Zernike expansion `g_{nlm}`

    @note The radon transform coefficients are given by `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)`. The Radon transform in coordinate space is given by `g_{nlm}P_n(w + dot(n,v))Y_{lm}(n)`, where `n` represents a unit vector and `v` is the boost vector.
*/
void radon_transform(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept;

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param exp Zernike expansion

    @note This function expects `f_{nlm}` to be contained in the lower portion of `exp`, such that its order is `exp.order() - 2`, and the rest of the values to be zero.

    @note The radon transform coefficients are given by `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)`. The Radon transform in coordinate space is given by `g_{nlm}P_n(w + dot(n,v))Y_{lm}(n)`, where `n` represents a unit vector and `v` is the boost vector.
*/
void radon_transform_inplace(
    ZernikeExpansionSpan<std::array<double, 2>> exp) noexcept;

} // namespace zebra
} // namespace zdm