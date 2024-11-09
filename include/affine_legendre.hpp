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

#include <vector>
#include <algorithm>

#include "zest/triangle_layout.hpp"
#include "zest/md_span.hpp"

#include "legendre.hpp"

namespace zdm
{

namespace zebra
{

/*
Recursion for expanding a Legendre function with affine transformed argument in terms of Legendre functions.

A Legendre polynomial `P_n(a + bx)` with an affinely transformed argument is a polynomial in `x`, and can therefore be expanded in terms of Legendre polynomials in `x`:
```txt
P_n(a + bx) = A_n0(a, b)P_0(x) + A_n1(a, b)P_1(x) + ... + A_nn(a, b)P_n(x)
```
This recursion generates the coefficients `A_nl(a, b)` for `0 <= l <= n <= lmax` for given parameters `a` and `b`.
*/
class AffineLegendreRecursion
{
public:
    AffineLegendreRecursion() = default;
    explicit AffineLegendreRecursion(std::size_t max_order);

    [[nodiscard]] std::size_t max_order() const noexcept { return m_max_order; }

    /*
    Expand the number of cached recursion coefficients up to `max_order`.
    */
    void expand(std::size_t max_order);

    /*
    Evaluate recursion of Legendre polynomials with affine transformed argument `y = shift + scale*x`.
    */
    void evaluate_affine(
        zest::TriangleSpan<double, zest::TriangleLayout> expansion, double shift, double scale);

    /*
    Evaluate recursion of Legendre polynomials with shifted argument `y = shift + x`.
    */
    void evaluate_shifted(
        zest::TriangleSpan<double, zest::TriangleLayout> expansion, double shift);

    /*
    Evaluate recursion of Legendre polynomials with scaled argument `y = scale*x`.
    */
    void evaluate_scaled(
        zest::TriangleSpan<double, zest::TriangleLayout> expansion, double scale);

private:
    std::vector<double> m_a;
    std::vector<double> m_b;
    std::vector<double> m_c;
    std::vector<double> m_d;
    std::size_t m_max_order;
};

} // namespace zebra
} // namespace zdm