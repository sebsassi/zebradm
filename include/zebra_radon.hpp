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

#include <zest/zernike_expansion.hpp>

#include "types.hpp"

namespace zdm::zebra
{

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param in input Zernike expansion coefficients
    @param out output Radon expansion coefficients

    Given the Zernike expansion coefficients \f$ f_{nlm} \f$ of a function \f$ f(\vec{x}) \f$, this
    computes the coefficients in the corresponding expansion of the offset Radon transform
    @f[
    \mathcal{R}[f](w,\hat{n}) = \sum_{nlm}g_{gnlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).
    @f]
    These coefficients are given by the formula
    @f[
    g_{nlm} = \frac{f_{nlm}}{2n + 3} - \frac{f_{n-2,lm}}{2n - 1}.
    @f]
*/
void radon_transform(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept;

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param exp Zernike expansion coefficients

    Given the Zernike expansion coefficients \f$ f_{nlm} \f$ of a function \f$ f(\vec{x}) \f$, this
    computes the coefficients in the corresponding expansion of the offset Radon transform
    @f[
    \mathcal{R}[f](w,\hat{n}) = \sum_{nlm}g_{gnlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).
    @f]
    These coefficients are given by the formula
    @f[
    g_{nlm} = \frac{f_{nlm}}{2n + 3} - \frac{f_{n-2,lm}}{2n - 1}.
    @f]
*/
void radon_transform_inplace(
    ZernikeExpansionSpan<std::array<double, 2>> exp) noexcept;

} // namespace zdm::zebra
