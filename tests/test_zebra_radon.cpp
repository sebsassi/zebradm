/*
Copyright (c) 2024-2026 Sebastian Sassi

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
#include "zebra_radon.hpp"

#include <cassert>
#include <print>


#include "radon_util.hpp"

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

bool test_zebra_radon_accepts_order(std::size_t order)
{
    zdm::ZernikeExpansion in{order};
    zdm::ZernikeExpansion out{order + 2};
    zdm::zebra::radon_transform(in, out);
    return true;
}

bool test_zebra_radon_is_correct_to_order_5()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 5;
    zdm::ZernikeExpansion in{order};
    in[0, 0, 0, 0] = 1.0;
    in[0, 0, 0, 1] = -1.0;
    in[1, 1, 0, 0] = 2.0;
    in[1, 1, 0, 1] = -2.0;
    in[1, 1, 1, 0] = 3.0;
    in[1, 1, 1, 1] = -3.0;
    in[2, 0, 0, 0] = 4.0;
    in[2, 0, 0, 1] = -4.0;
    in[2, 2, 0, 0] = 5.0;
    in[2, 2, 0, 1] = -5.0;
    in[2, 2, 1, 0] = 6.0;
    in[2, 2, 1, 1] = -6.0;
    in[2, 2, 2, 0] = 7.0;
    in[2, 2, 2, 1] = -7.0;
    in[3, 1, 0, 0] = 8.0;
    in[3, 1, 0, 1] = -8.0;
    in[3, 1, 1, 0] = 9.0;
    in[3, 1, 1, 1] = -9.0;
    in[3, 3, 0, 0] = 10.0;
    in[3, 3, 0, 1] = -10.0;
    in[3, 3, 1, 0] = 11.0;
    in[3, 3, 1, 1] = -11.0;
    in[3, 3, 2, 0] = 12.0;
    in[3, 3, 2, 1] = -12.0;
    in[3, 3, 3, 0] = 13.0;
    in[3, 3, 3, 1] = -13.0;
    in[4, 0, 0, 0] = 14.0;
    in[4, 0, 0, 1] = -14.0;
    in[4, 2, 0, 0] = 15.0;
    in[4, 2, 0, 1] = -15.0;
    in[4, 2, 1, 0] = 16.0;
    in[4, 2, 1, 1] = -16.0;
    in[4, 2, 2, 0] = 17.0;
    in[4, 2, 2, 1] = -17.0;
    in[4, 4, 0, 0] = 18.0;
    in[4, 4, 0, 1] = -18.0;
    in[4, 4, 1, 0] = 19.0;
    in[4, 4, 1, 1] = -19.0;
    in[4, 4, 2, 0] = 20.0;
    in[4, 4, 2, 1] = -20.0;
    in[4, 4, 3, 0] = 21.0;
    in[4, 4, 3, 1] = -21.0;
    in[4, 4, 4, 0] = 22.0;
    in[4, 4, 4, 1] = -22.0;

    zdm::ZernikeExpansion out_ref{order + 2};
    out_ref[0, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(0)*in[0, 0, 0, 0];
    out_ref[0, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(0)*in[0, 0, 0, 1];
    out_ref[1, 1, 0, 0] = zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 0, 0];
    out_ref[1, 1, 0, 1] = zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 0, 1];
    out_ref[1, 1, 1, 0] = zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 1, 0];
    out_ref[1, 1, 1, 1] = zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 1, 1];
    out_ref[2, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 0, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(0)*in[0, 0, 0, 0];
    out_ref[2, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 0, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(0)*in[0, 0, 0, 1];
    out_ref[2, 2, 0, 0] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 0, 0];
    out_ref[2, 2, 0, 1] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 0, 1];
    out_ref[2, 2, 1, 0] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 1, 0];
    out_ref[2, 2, 1, 1] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 1, 1];
    out_ref[2, 2, 2, 0] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 2, 0];
    out_ref[2, 2, 2, 1] = zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 2, 1];
    out_ref[3, 1, 0, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 0, 0];
    out_ref[3, 1, 0, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 0, 1];
    out_ref[3, 1, 1, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 1, 0]
                        - zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 1, 0];
    out_ref[3, 1, 1, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 1, 1]
                        - zdm::util::geg_rec_coeff<norm>(1)*in[1, 1, 1, 1];
    out_ref[3, 3, 0, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 0, 0];
    out_ref[3, 3, 0, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 0, 1];
    out_ref[3, 3, 1, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 1, 0];
    out_ref[3, 3, 1, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 1, 1];
    out_ref[3, 3, 2, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 2, 0];
    out_ref[3, 3, 2, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 2, 1];
    out_ref[3, 3, 3, 0] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 3, 0];
    out_ref[3, 3, 3, 1] = zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 3, 1];
    out_ref[4, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 0, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 0, 0, 0];
    out_ref[4, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 0, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 0, 0, 1];
    out_ref[4, 2, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 0, 0];
    out_ref[4, 2, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 0, 1];
    out_ref[4, 2, 1, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 1, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 1, 0];
    out_ref[4, 2, 1, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 1, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 1, 1];
    out_ref[4, 2, 2, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 2, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 2, 0];
    out_ref[4, 2, 2, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 2, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*in[2, 2, 2, 1];
    out_ref[4, 4, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 0, 0];
    out_ref[4, 4, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 0, 1];
    out_ref[4, 4, 1, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 1, 0];
    out_ref[4, 4, 1, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 1, 1];
    out_ref[4, 4, 2, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 2, 0];
    out_ref[4, 4, 2, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 2, 1];
    out_ref[4, 4, 3, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 3, 0];
    out_ref[4, 4, 3, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 3, 1];
    out_ref[4, 4, 4, 0] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 4, 0];
    out_ref[4, 4, 4, 1] = zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 4, 1];
    out_ref[5, 1, 0, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 0, 0];
    out_ref[5, 1, 0, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 0, 1];
    out_ref[5, 1, 1, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 1, 0];
    out_ref[5, 1, 1, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 1, 1, 1];
    out_ref[5, 3, 0, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 0, 0];
    out_ref[5, 3, 0, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 0, 1];
    out_ref[5, 3, 1, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 1, 0];
    out_ref[5, 3, 1, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 1, 1];
    out_ref[5, 3, 2, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 2, 0];
    out_ref[5, 3, 2, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 2, 1];
    out_ref[5, 3, 3, 0] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 3, 0];
    out_ref[5, 3, 3, 1] = -zdm::util::geg_rec_coeff<norm>(3)*in[3, 3, 3, 1];
    out_ref[5, 5, 0, 0] = 0.0;
    out_ref[5, 5, 0, 1] = 0.0;
    out_ref[5, 5, 1, 0] = 0.0;
    out_ref[5, 5, 1, 1] = 0.0;
    out_ref[5, 5, 2, 0] = 0.0;
    out_ref[5, 5, 2, 1] = 0.0;
    out_ref[5, 5, 3, 0] = 0.0;
    out_ref[5, 5, 3, 1] = 0.0;
    out_ref[5, 5, 4, 0] = 0.0;
    out_ref[5, 5, 4, 1] = 0.0;
    out_ref[5, 5, 5, 0] = 0.0;
    out_ref[5, 5, 5, 1] = 0.0;
    out_ref[6, 0, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 0, 0, 0];
    out_ref[6, 0, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 0, 0, 1];
    out_ref[6, 2, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 0, 0];
    out_ref[6, 2, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 0, 1];
    out_ref[6, 2, 1, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 1, 0];
    out_ref[6, 2, 1, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 1, 1];
    out_ref[6, 2, 2, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 2, 0];
    out_ref[6, 2, 2, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 2, 2, 1];
    out_ref[6, 4, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 0, 0];
    out_ref[6, 4, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 0, 1];
    out_ref[6, 4, 1, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 1, 0];
    out_ref[6, 4, 1, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 1, 1];
    out_ref[6, 4, 2, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 2, 0];
    out_ref[6, 4, 2, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 2, 1];
    out_ref[6, 4, 3, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 3, 0];
    out_ref[6, 4, 3, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 3, 1];
    out_ref[6, 4, 4, 0] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 4, 0];
    out_ref[6, 4, 4, 1] = -zdm::util::geg_rec_coeff<norm>(4)*in[4, 4, 4, 1];
    out_ref[6, 6, 0, 0] = 0.0;
    out_ref[6, 6, 0, 1] = 0.0;
    out_ref[6, 6, 1, 0] = 0.0;
    out_ref[6, 6, 1, 1] = 0.0;
    out_ref[6, 6, 2, 0] = 0.0;
    out_ref[6, 6, 2, 1] = 0.0;
    out_ref[6, 6, 3, 0] = 0.0;
    out_ref[6, 6, 3, 1] = 0.0;
    out_ref[6, 6, 4, 0] = 0.0;
    out_ref[6, 6, 4, 1] = 0.0;
    out_ref[6, 6, 5, 0] = 0.0;
    out_ref[6, 6, 5, 1] = 0.0;
    out_ref[6, 6, 6, 0] = 0.0;
    out_ref[6, 6, 6, 1] = 0.0;

    zdm::ZernikeExpansion out{order + 2};

    zdm::zebra::radon_transform(in, out);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n = 0; n < order + 2; ++n)
    {
        for (std::size_t l = n & 1; l <= n; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                success = success && (is_close(out[n, l, m, 0], out_ref[n, l, m, 0], tol));
                success = success && (is_close(out[n, l, m, 1], out_ref[n, l, m, 1], tol));
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n < order + 2; ++n)
        {
            for (std::size_t l = n & 1; l <= n; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::println("({}, {}, {}): [{}, {}] [{}, {}]", n, l, m, out[n, l, m, 0], out[n, l, m, 1], out_ref[n, l, m, 0], out_ref[n, l, m, 1]);
            }
        }
    }

    return success;
}



bool test_inplace_zebra_radon_is_correct_to_order_5()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 5;
    zdm::ZernikeExpansion exp{order + 2};
    exp[0, 0, 0, 0] = 1.0;
    exp[0, 0, 0, 1] = -1.0;
    exp[1, 1, 0, 0] = 2.0;
    exp[1, 1, 0, 1] = -2.0;
    exp[1, 1, 1, 0] = 3.0;
    exp[1, 1, 1, 1] = -3.0;
    exp[2, 0, 0, 0] = 4.0;
    exp[2, 0, 0, 1] = -4.0;
    exp[2, 2, 0, 0] = 5.0;
    exp[2, 2, 0, 1] = -5.0;
    exp[2, 2, 1, 0] = 6.0;
    exp[2, 2, 1, 1] = -6.0;
    exp[2, 2, 2, 0] = 7.0;
    exp[2, 2, 2, 1] = -7.0;
    exp[3, 1, 0, 0] = 8.0;
    exp[3, 1, 0, 1] = -8.0;
    exp[3, 1, 1, 0] = 9.0;
    exp[3, 1, 1, 1] = -9.0;
    exp[3, 3, 0, 0] = 10.0;
    exp[3, 3, 0, 1] = -10.0;
    exp[3, 3, 1, 0] = 11.0;
    exp[3, 3, 1, 1] = -11.0;
    exp[3, 3, 2, 0] = 12.0;
    exp[3, 3, 2, 1] = -12.0;
    exp[3, 3, 3, 0] = 13.0;
    exp[3, 3, 3, 1] = -13.0;
    exp[4, 0, 0, 0] = 14.0;
    exp[4, 0, 0, 1] = -14.0;
    exp[4, 2, 0, 0] = 15.0;
    exp[4, 2, 0, 1] = -15.0;
    exp[4, 2, 1, 0] = 16.0;
    exp[4, 2, 1, 1] = -16.0;
    exp[4, 2, 2, 0] = 17.0;
    exp[4, 2, 2, 1] = -17.0;
    exp[4, 4, 0, 0] = 18.0;
    exp[4, 4, 0, 1] = -18.0;
    exp[4, 4, 1, 0] = 19.0;
    exp[4, 4, 1, 1] = -19.0;
    exp[4, 4, 2, 0] = 20.0;
    exp[4, 4, 2, 1] = -20.0;
    exp[4, 4, 3, 0] = 21.0;
    exp[4, 4, 3, 1] = -21.0;
    exp[4, 4, 4, 0] = 22.0;
    exp[4, 4, 4, 1] = -22.0;

    zdm::ZernikeExpansion out_ref{order + 2};
    out_ref[0, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(0)*exp[0, 0, 0, 0];
    out_ref[0, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(0)*exp[0, 0, 0, 1];
    out_ref[1, 1, 0, 0] = zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 0, 0];
    out_ref[1, 1, 0, 1] = zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 0, 1];
    out_ref[1, 1, 1, 0] = zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 1, 0];
    out_ref[1, 1, 1, 1] = zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 1, 1];
    out_ref[2, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 0, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(0)*exp[0, 0, 0, 0];
    out_ref[2, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 0, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(0)*exp[0, 0, 0, 1];
    out_ref[2, 2, 0, 0] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 0, 0];
    out_ref[2, 2, 0, 1] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 0, 1];
    out_ref[2, 2, 1, 0] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 1, 0];
    out_ref[2, 2, 1, 1] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 1, 1];
    out_ref[2, 2, 2, 0] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 2, 0];
    out_ref[2, 2, 2, 1] = zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 2, 1];
    out_ref[3, 1, 0, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 0, 0];
    out_ref[3, 1, 0, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 0, 1];
    out_ref[3, 1, 1, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 1, 0]
                        - zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 1, 0];
    out_ref[3, 1, 1, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 1, 1]
                        - zdm::util::geg_rec_coeff<norm>(1)*exp[1, 1, 1, 1];
    out_ref[3, 3, 0, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 0, 0];
    out_ref[3, 3, 0, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 0, 1];
    out_ref[3, 3, 1, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 1, 0];
    out_ref[3, 3, 1, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 1, 1];
    out_ref[3, 3, 2, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 2, 0];
    out_ref[3, 3, 2, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 2, 1];
    out_ref[3, 3, 3, 0] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 3, 0];
    out_ref[3, 3, 3, 1] = zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 3, 1];
    out_ref[4, 0, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 0, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 0, 0, 0];
    out_ref[4, 0, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 0, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 0, 0, 1];
    out_ref[4, 2, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 0, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 0, 0];
    out_ref[4, 2, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 0, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 0, 1];
    out_ref[4, 2, 1, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 1, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 1, 0];
    out_ref[4, 2, 1, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 1, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 1, 1];
    out_ref[4, 2, 2, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 2, 0]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 2, 0];
    out_ref[4, 2, 2, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 2, 1]
                        - zdm::util::geg_rec_coeff<norm>(2)*exp[2, 2, 2, 1];
    out_ref[4, 4, 0, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 0, 0];
    out_ref[4, 4, 0, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 0, 1];
    out_ref[4, 4, 1, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 1, 0];
    out_ref[4, 4, 1, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 1, 1];
    out_ref[4, 4, 2, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 2, 0];
    out_ref[4, 4, 2, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 2, 1];
    out_ref[4, 4, 3, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 3, 0];
    out_ref[4, 4, 3, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 3, 1];
    out_ref[4, 4, 4, 0] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 4, 0];
    out_ref[4, 4, 4, 1] = zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 4, 1];
    out_ref[5, 1, 0, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 0, 0];
    out_ref[5, 1, 0, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 0, 1];
    out_ref[5, 1, 1, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 1, 0];
    out_ref[5, 1, 1, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 1, 1, 1];
    out_ref[5, 3, 0, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 0, 0];
    out_ref[5, 3, 0, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 0, 1];
    out_ref[5, 3, 1, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 1, 0];
    out_ref[5, 3, 1, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 1, 1];
    out_ref[5, 3, 2, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 2, 0];
    out_ref[5, 3, 2, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 2, 1];
    out_ref[5, 3, 3, 0] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 3, 0];
    out_ref[5, 3, 3, 1] = -zdm::util::geg_rec_coeff<norm>(3)*exp[3, 3, 3, 1];
    out_ref[5, 5, 0, 0] = 0.0;
    out_ref[5, 5, 0, 1] = 0.0;
    out_ref[5, 5, 1, 0] = 0.0;
    out_ref[5, 5, 1, 1] = 0.0;
    out_ref[5, 5, 2, 0] = 0.0;
    out_ref[5, 5, 2, 1] = 0.0;
    out_ref[5, 5, 3, 0] = 0.0;
    out_ref[5, 5, 3, 1] = 0.0;
    out_ref[5, 5, 4, 0] = 0.0;
    out_ref[5, 5, 4, 1] = 0.0;
    out_ref[5, 5, 5, 0] = 0.0;
    out_ref[5, 5, 5, 1] = 0.0;
    out_ref[6, 0, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 0, 0, 0];
    out_ref[6, 0, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 0, 0, 1];
    out_ref[6, 2, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 0, 0];
    out_ref[6, 2, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 0, 1];
    out_ref[6, 2, 1, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 1, 0];
    out_ref[6, 2, 1, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 1, 1];
    out_ref[6, 2, 2, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 2, 0];
    out_ref[6, 2, 2, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 2, 2, 1];
    out_ref[6, 4, 0, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 0, 0];
    out_ref[6, 4, 0, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 0, 1];
    out_ref[6, 4, 1, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 1, 0];
    out_ref[6, 4, 1, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 1, 1];
    out_ref[6, 4, 2, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 2, 0];
    out_ref[6, 4, 2, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 2, 1];
    out_ref[6, 4, 3, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 3, 0];
    out_ref[6, 4, 3, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 3, 1];
    out_ref[6, 4, 4, 0] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 4, 0];
    out_ref[6, 4, 4, 1] = -zdm::util::geg_rec_coeff<norm>(4)*exp[4, 4, 4, 1];
    out_ref[6, 6, 0, 0] = 0.0;
    out_ref[6, 6, 0, 1] = 0.0;
    out_ref[6, 6, 1, 0] = 0.0;
    out_ref[6, 6, 1, 1] = 0.0;
    out_ref[6, 6, 2, 0] = 0.0;
    out_ref[6, 6, 2, 1] = 0.0;
    out_ref[6, 6, 3, 0] = 0.0;
    out_ref[6, 6, 3, 1] = 0.0;
    out_ref[6, 6, 4, 0] = 0.0;
    out_ref[6, 6, 4, 1] = 0.0;
    out_ref[6, 6, 5, 0] = 0.0;
    out_ref[6, 6, 5, 1] = 0.0;
    out_ref[6, 6, 6, 0] = 0.0;
    out_ref[6, 6, 6, 1] = 0.0;

    zdm::zebra::radon_transform_inplace(exp);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n = 0; n < order + 2; ++n)
    {
        for (std::size_t l = n & 1; l <= n; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                success = success && (is_close(exp[n, l, m, 0], out_ref[n, l, m, 0], tol));
                success = success && (is_close(exp[n, l, m, 1], out_ref[n, l, m, 1], tol));
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n < order + 2; ++n)
        {
            for (std::size_t l = n & 1; l <= n; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::println("({}, {}, {}): [{}, {}] [{}, {}]", n, l, m, exp[n, l, m, 0], exp[n, l, m, 1], out_ref[n, l, m, 0], out_ref[n, l, m, 1]);
            }
        }
    }

    return success;
}

} // namespace

int main()
{
    assert(test_zebra_radon_accepts_order(0));
    assert(test_zebra_radon_accepts_order(1));
    assert(test_zebra_radon_accepts_order(2));
    assert(test_zebra_radon_accepts_order(3));
    assert(test_zebra_radon_is_correct_to_order_5());
    assert(test_inplace_zebra_radon_is_correct_to_order_5());
}
