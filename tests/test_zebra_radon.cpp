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

#include <cassert>
#include <print>

#include "radon_util.hpp"
#include "zebra_radon.hpp"

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

bool test_zebra_radon_accepts_order(std::size_t order)
{
    zdm::ZernikeExpansion<double> in{order};
    zdm::ZernikeExpansion<double> out{order + 2};
    zdm::zebra::radon_transform(in, out);
    return true;
}

bool test_zebra_radon_is_correct_to_order_5()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 5;
    zdm::ZernikeExpansion<double> in{order};
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

    zdm::ZernikeExpansion<double> out_ref{order + 2};
    out_ref[0, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(0)*in[0, 0, 0, 0];
    out_ref[0, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(0)*in[0, 0, 0, 1];
    out_ref[1, 1, 0, 0] = zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 0, 0];
    out_ref[1, 1, 0, 1] = zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 0, 1];
    out_ref[1, 1, 1, 0] = zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 1, 0];
    out_ref[1, 1, 1, 1] = zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 1, 1];
    out_ref[2, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 0, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(0)*in[0, 0, 0, 0];
    out_ref[2, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 0, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(0)*in[0, 0, 0, 1];
    out_ref[2, 2, 0, 0] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 0, 0];
    out_ref[2, 2, 0, 1] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 0, 1];
    out_ref[2, 2, 1, 0] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 1, 0];
    out_ref[2, 2, 1, 1] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 1, 1];
    out_ref[2, 2, 2, 0] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 2, 0];
    out_ref[2, 2, 2, 1] = zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 2, 1];
    out_ref[3, 1, 0, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 0, 0];
    out_ref[3, 1, 0, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 0, 1];
    out_ref[3, 1, 1, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 1, 0]
                        - zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 1, 0];
    out_ref[3, 1, 1, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 1, 1]
                        - zdm::util::zernike_radon_coeff<norm>(1)*in[1, 1, 1, 1];
    out_ref[3, 3, 0, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 0, 0];
    out_ref[3, 3, 0, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 0, 1];
    out_ref[3, 3, 1, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 1, 0];
    out_ref[3, 3, 1, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 1, 1];
    out_ref[3, 3, 2, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 2, 0];
    out_ref[3, 3, 2, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 2, 1];
    out_ref[3, 3, 3, 0] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 3, 0];
    out_ref[3, 3, 3, 1] = zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 3, 1];
    out_ref[4, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 0, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 0, 0, 0];
    out_ref[4, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 0, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 0, 0, 1];
    out_ref[4, 2, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 0, 0];
    out_ref[4, 2, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 0, 1];
    out_ref[4, 2, 1, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 1, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 1, 0];
    out_ref[4, 2, 1, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 1, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 1, 1];
    out_ref[4, 2, 2, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 2, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 2, 0];
    out_ref[4, 2, 2, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 2, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2, 2, 2, 1];
    out_ref[4, 4, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 0, 0];
    out_ref[4, 4, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 0, 1];
    out_ref[4, 4, 1, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 1, 0];
    out_ref[4, 4, 1, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 1, 1];
    out_ref[4, 4, 2, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 2, 0];
    out_ref[4, 4, 2, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 2, 1];
    out_ref[4, 4, 3, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 3, 0];
    out_ref[4, 4, 3, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 3, 1];
    out_ref[4, 4, 4, 0] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 4, 0];
    out_ref[4, 4, 4, 1] = zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 4, 1];
    out_ref[5, 1, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 0, 0];
    out_ref[5, 1, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 0, 1];
    out_ref[5, 1, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 1, 0];
    out_ref[5, 1, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 1, 1, 1];
    out_ref[5, 3, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 0, 0];
    out_ref[5, 3, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 0, 1];
    out_ref[5, 3, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 1, 0];
    out_ref[5, 3, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 1, 1];
    out_ref[5, 3, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 2, 0];
    out_ref[5, 3, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 2, 1];
    out_ref[5, 3, 3, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 3, 0];
    out_ref[5, 3, 3, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*in[3, 3, 3, 1];
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
    out_ref[6, 0, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 0, 0, 0];
    out_ref[6, 0, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 0, 0, 1];
    out_ref[6, 2, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 0, 0];
    out_ref[6, 2, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 0, 1];
    out_ref[6, 2, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 1, 0];
    out_ref[6, 2, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 1, 1];
    out_ref[6, 2, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 2, 0];
    out_ref[6, 2, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 2, 2, 1];
    out_ref[6, 4, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 0, 0];
    out_ref[6, 4, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 0, 1];
    out_ref[6, 4, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 1, 0];
    out_ref[6, 4, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 1, 1];
    out_ref[6, 4, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 2, 0];
    out_ref[6, 4, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 2, 1];
    out_ref[6, 4, 3, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 3, 0];
    out_ref[6, 4, 3, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 3, 1];
    out_ref[6, 4, 4, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 4, 0];
    out_ref[6, 4, 4, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*in[4, 4, 4, 1];
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

    zdm::ZernikeExpansion<double> out{order + 2};

    zdm::zebra::radon_transform(in, out);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : out.indices())
    {
        auto out_n = out[n];
        auto out_ref_n = out_ref[n];
        for (std::size_t l : out_n.indices())
        {
            auto out_nl = out_n[l];
            auto out_ref_nl = out_ref_n[l];
            for (std::size_t m : out_nl.indices())
            {
                success = success && (is_close(out_nl[m, 0], out_ref_nl[m, 0], tol));
                success = success && (is_close(out_nl[m, 1], out_ref_nl[m, 1], tol));
            }
        }
    }

    if (!success)
    {
        for (std::size_t n : out.indices())
        {
            auto out_n = out[n];
            auto out_ref_n = out_ref[n];
            for (std::size_t l : out_n.indices())
            {
                auto out_nl = out_n[l];
                auto out_ref_nl = out_ref_n[l];
                for (std::size_t m : out_nl.indices())
                    std::println("({}, {}, {}): [{}, {}] [{}, {}]", n, l, m, out_nl[m, 0], out_nl[m, 1], out_ref_nl[m, 0], out_ref_nl[m, 1]);
            }
        }
    }

    return success;
}

bool test_inplace_zebra_radon_is_correct_to_order_5()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 5;
    zdm::ZernikeExpansion<double> exp{order + 2};
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

    zdm::ZernikeExpansion<double> out_ref{order + 2};
    out_ref[0, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(0)*exp[0, 0, 0, 0];
    out_ref[0, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(0)*exp[0, 0, 0, 1];
    out_ref[1, 1, 0, 0] = zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 0, 0];
    out_ref[1, 1, 0, 1] = zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 0, 1];
    out_ref[1, 1, 1, 0] = zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 1, 0];
    out_ref[1, 1, 1, 1] = zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 1, 1];
    out_ref[2, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 0, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(0)*exp[0, 0, 0, 0];
    out_ref[2, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 0, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(0)*exp[0, 0, 0, 1];
    out_ref[2, 2, 0, 0] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 0, 0];
    out_ref[2, 2, 0, 1] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 0, 1];
    out_ref[2, 2, 1, 0] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 1, 0];
    out_ref[2, 2, 1, 1] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 1, 1];
    out_ref[2, 2, 2, 0] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 2, 0];
    out_ref[2, 2, 2, 1] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 2, 1];
    out_ref[3, 1, 0, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 0, 0];
    out_ref[3, 1, 0, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 0, 1];
    out_ref[3, 1, 1, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 1, 0]
                        - zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 1, 0];
    out_ref[3, 1, 1, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 1, 1]
                        - zdm::util::zernike_radon_coeff<norm>(1)*exp[1, 1, 1, 1];
    out_ref[3, 3, 0, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 0, 0];
    out_ref[3, 3, 0, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 0, 1];
    out_ref[3, 3, 1, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 1, 0];
    out_ref[3, 3, 1, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 1, 1];
    out_ref[3, 3, 2, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 2, 0];
    out_ref[3, 3, 2, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 2, 1];
    out_ref[3, 3, 3, 0] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 3, 0];
    out_ref[3, 3, 3, 1] = zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 3, 1];
    out_ref[4, 0, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 0, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 0, 0, 0];
    out_ref[4, 0, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 0, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 0, 0, 1];
    out_ref[4, 2, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 0, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 0, 0];
    out_ref[4, 2, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 0, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 0, 1];
    out_ref[4, 2, 1, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 1, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 1, 0];
    out_ref[4, 2, 1, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 1, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 1, 1];
    out_ref[4, 2, 2, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 2, 0]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 2, 0];
    out_ref[4, 2, 2, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 2, 1]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2, 2, 2, 1];
    out_ref[4, 4, 0, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 0, 0];
    out_ref[4, 4, 0, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 0, 1];
    out_ref[4, 4, 1, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 1, 0];
    out_ref[4, 4, 1, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 1, 1];
    out_ref[4, 4, 2, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 2, 0];
    out_ref[4, 4, 2, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 2, 1];
    out_ref[4, 4, 3, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 3, 0];
    out_ref[4, 4, 3, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 3, 1];
    out_ref[4, 4, 4, 0] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 4, 0];
    out_ref[4, 4, 4, 1] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 4, 1];
    out_ref[5, 1, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 0, 0];
    out_ref[5, 1, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 0, 1];
    out_ref[5, 1, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 1, 0];
    out_ref[5, 1, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 1, 1, 1];
    out_ref[5, 3, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 0, 0];
    out_ref[5, 3, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 0, 1];
    out_ref[5, 3, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 1, 0];
    out_ref[5, 3, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 1, 1];
    out_ref[5, 3, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 2, 0];
    out_ref[5, 3, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 2, 1];
    out_ref[5, 3, 3, 0] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 3, 0];
    out_ref[5, 3, 3, 1] = -zdm::util::zernike_radon_coeff<norm>(3)*exp[3, 3, 3, 1];
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
    out_ref[6, 0, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 0, 0, 0];
    out_ref[6, 0, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 0, 0, 1];
    out_ref[6, 2, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 0, 0];
    out_ref[6, 2, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 0, 1];
    out_ref[6, 2, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 1, 0];
    out_ref[6, 2, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 1, 1];
    out_ref[6, 2, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 2, 0];
    out_ref[6, 2, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 2, 2, 1];
    out_ref[6, 4, 0, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 0, 0];
    out_ref[6, 4, 0, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 0, 1];
    out_ref[6, 4, 1, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 1, 0];
    out_ref[6, 4, 1, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 1, 1];
    out_ref[6, 4, 2, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 2, 0];
    out_ref[6, 4, 2, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 2, 1];
    out_ref[6, 4, 3, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 3, 0];
    out_ref[6, 4, 3, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 3, 1];
    out_ref[6, 4, 4, 0] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 4, 0];
    out_ref[6, 4, 4, 1] = -zdm::util::zernike_radon_coeff<norm>(4)*exp[4, 4, 4, 1];
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
    for (std::size_t n : exp.indices())
    {
        auto exp_n = exp[n];
        auto out_ref_n = out_ref[n];
        for (std::size_t l : exp_n.indices())
        {
            auto exp_nl = exp_n[l];
            auto out_ref_nl = out_ref_n[l];
            for (std::size_t m : exp_nl.indices())
            {
                success = success && (is_close(exp_nl[m, 0], out_ref_nl[m, 0], tol));
                success = success && (is_close(exp_nl[m, 1], out_ref_nl[m, 1], tol));
            }
        }
    }

    if (!success)
    {
        for (std::size_t n : exp.indices())
        {
            auto exp_n = exp[n];
            auto out_ref_n = out_ref[n];
            for (std::size_t l : exp_n.indices())
            {
                auto exp_nl = exp_n[l];
                auto out_ref_nl = out_ref_n[l];
                for (std::size_t m : exp_nl.indices())
                    std::println("({}, {}, {}): [{}, {}] [{}, {}]", n, l, m, exp_nl[m, 0], exp_nl[m, 1], out_ref_nl[m, 0], out_ref_nl[m, 1]);
            }
        }
    }

    return success;
}

bool test_isotropic_zebra_radon_is_correct_to_order_9()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 9;
    zdm::IsotropicZernikeExpansion<double> in{order};
    in[0] = 1.0;
    in[2] = 2.0;
    in[4] = 3.0;
    in[6] = 4.0;
    in[8] = 5.0;

    zdm::IsotropicZernikeExpansion<double> out_ref{order + 2};
    out_ref[0] = zdm::util::zernike_radon_coeff<norm>(0)*in[0];
    out_ref[2] = zdm::util::zernike_radon_coeff<norm>(2)*in[2]
                        - zdm::util::zernike_radon_coeff<norm>(0)*in[0];
    out_ref[4] = zdm::util::zernike_radon_coeff<norm>(4)*in[4]
                        - zdm::util::zernike_radon_coeff<norm>(2)*in[2];
    out_ref[6] = zdm::util::zernike_radon_coeff<norm>(6)*in[6]
                        - zdm::util::zernike_radon_coeff<norm>(4)*in[4];
    out_ref[8] = zdm::util::zernike_radon_coeff<norm>(8)*in[8]
                        - zdm::util::zernike_radon_coeff<norm>(6)*in[6];
    out_ref[10] = -zdm::util::zernike_radon_coeff<norm>(8)*in[8];

    zdm::IsotropicZernikeExpansion<double> out{order + 2};

    zdm::zebra::radon_transform(in, out);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : out.indices())
        success = success && is_close(out[n], out_ref[n], tol);

    if (!success)
    {
        for (std::size_t n : out.indices())
            std::println("{}: {} {}", n, out[n], out_ref[n]);
    }

    return success;
}

bool test_inplace_isotropic_zebra_radon_is_correct_to_order_9()
{
    static constexpr zest::zt::ZernikeNorm norm = zest::zt::ZernikeNorm::normed;
    constexpr std::size_t order = 9;
    zdm::IsotropicZernikeExpansion<double> exp{order + 2};
    exp[0] = 1.0;
    exp[2] = 2.0;
    exp[4] = 3.0;
    exp[6] = 4.0;
    exp[8] = 5.0;

    zdm::IsotropicZernikeExpansion<double> out_ref{order + 2};
    out_ref[0] = zdm::util::zernike_radon_coeff<norm>(0)*exp[0];
    out_ref[2] = zdm::util::zernike_radon_coeff<norm>(2)*exp[2]
                        - zdm::util::zernike_radon_coeff<norm>(0)*exp[0];
    out_ref[4] = zdm::util::zernike_radon_coeff<norm>(4)*exp[4]
                        - zdm::util::zernike_radon_coeff<norm>(2)*exp[2];
    out_ref[6] = zdm::util::zernike_radon_coeff<norm>(6)*exp[6]
                        - zdm::util::zernike_radon_coeff<norm>(4)*exp[4];
    out_ref[8] = zdm::util::zernike_radon_coeff<norm>(8)*exp[8]
                        - zdm::util::zernike_radon_coeff<norm>(6)*exp[6];
    out_ref[10] = -zdm::util::zernike_radon_coeff<norm>(8)*exp[8];

    zdm::zebra::radon_transform_inplace(exp);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : exp.indices())
        success = success && is_close(exp[n], out_ref[n], tol);

    if (!success)
    {
        for (std::size_t n : exp.indices())
            std::println("{}: {} {}", n, exp[n], out_ref[n]);
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

    assert(test_isotropic_zebra_radon_is_correct_to_order_9());
    assert(test_inplace_isotropic_zebra_radon_is_correct_to_order_9());
}
