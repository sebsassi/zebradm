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

#include <zest/zernike_conventions.hpp>

#include "radon_util.hpp"
#include "utility.hpp"

namespace zdm::zebra
{

void radon_transform(ZernikeSpan<const double> in, ZernikeSpan<double> out) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeSpan<const double>::shape_type::zernike_norm;

    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 <= out.order());

    if (in.order() == 0) return;

    const double coeff_0 = util::zernike_radon_coeff<zernike_norm>(0);
    out[0, 0, 0, 0] = coeff_0*in[0, 0, 0, 0];
    out[0, 0, 0, 1] = coeff_0*in[0, 0, 0, 1];

    if (in.order() > 1)
    {
        const double coeff_1 = util::zernike_radon_coeff<zernike_norm>(1);
        out[1, 1, 0, 0] = coeff_1*in[1, 1, 0, 0];
        out[1, 1, 0, 1] = coeff_1*in[1, 1, 0, 1];
        out[1, 1, 1, 0] = coeff_1*in[1, 1, 1, 0];
        out[1, 1, 1, 1] = coeff_1*in[1, 1, 1, 1];
    }
    else
    {
        out[2, 0, 0, 0] = -coeff_0*in[0, 0, 0, 0];
        out[2, 0, 0, 1] = -coeff_0*in[0, 0, 0, 1];
        out[2, 2, 0, 0] = 0.0;
        out[2, 2, 0, 1] = 0.0;
        out[2, 2, 1, 0] = 0.0;
        out[2, 2, 1, 1] = 0.0;
        out[2, 2, 2, 0] = 0.0;
        out[2, 2, 2, 1] = 0.0;
        return;
    }

    for (std::size_t n : in.indices())
    {
        auto out_n = out[n];
        auto in_n = in[n];
        auto in_nm2 = in[n - 2];

        const double coeff_n = util::zernike_radon_coeff<zernike_norm>(n);
        const double coeff_nm2 = -util::zernike_radon_coeff<zernike_norm>(n - 2);
        auto out_n_flat = out_n.flatten();
        auto in_n_flat = in_n.flatten();
        auto in_nm2_flat = in_nm2.flatten();

        const std::size_t out_n_size = out_n_flat.size();
        const std::size_t in_n_size = in_n_flat.size();
        const std::size_t in_nm2_size = in_nm2_flat.size();
        [[assume(out_n_size == in_n_size)]];
        [[assume(out_n_size >= in_nm2_size)]];
        util::linear_combination(out_n_flat, coeff_n, in_n_flat, coeff_nm2, in_nm2_flat);

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = coeff_n*in_n_n[m, 0];
            out_n_n[m, 1] = coeff_n*in_n_n[m, 1];
        }
    }

    for (std::size_t n : out.indices(in.order()))
    {
        auto out_n = out[n];
        auto in_nm2 = in[n - 2];

        const double coeff_nm2 = -util::zernike_radon_coeff<zernike_norm>(n - 2);
        auto out_n_flat = out_n.flatten();
        auto in_nm2_flat = in_nm2.flatten();

        const std::size_t out_n_size = out_n_flat.size();
        const std::size_t in_nm2_size = in_nm2_flat.size();
        [[assume(out_n_size >= in_nm2_size)]];
        util::mul(out_n_flat, coeff_nm2, in_nm2_flat);

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = 0.0;
            out_n_n[m, 1] = 0.0;
        }
    }
}

void radon_transform(
    ZernikeSpan<const double> in, ZernikeSpan<double> out,
    std::span<const double> zernike_radon_coeff) noexcept
{
    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 <= out.order());
    assert(zernike_radon_coeff.size() >= in.order());

    if (in.order() == 0) return;

    const double coeff_0 = zernike_radon_coeff[0];
    out[0, 0, 0, 0] = coeff_0*in[0, 0, 0, 0];
    out[0, 0, 0, 1] = coeff_0*in[0, 0, 0, 1];

    if (in.order() > 1)
    {
        const double coeff_1 = zernike_radon_coeff[1];
        out[1, 1, 0, 0] = coeff_1*in[1, 1, 0, 0];
        out[1, 1, 0, 1] = coeff_1*in[1, 1, 0, 1];
        out[1, 1, 1, 0] = coeff_1*in[1, 1, 1, 0];
        out[1, 1, 1, 1] = coeff_1*in[1, 1, 1, 1];
    }
    else
    {
        out[2, 0, 0, 0] = -coeff_0*in[0, 0, 0, 0];
        out[2, 0, 0, 1] = -coeff_0*in[0, 0, 0, 1];
        out[2, 2, 0, 0] = 0.0;
        out[2, 2, 0, 1] = 0.0;
        out[2, 2, 1, 0] = 0.0;
        out[2, 2, 1, 1] = 0.0;
        out[2, 2, 2, 0] = 0.0;
        out[2, 2, 2, 1] = 0.0;
        return;
    }

    for (std::size_t n : in.indices())
    {
        auto out_n = out[n];
        auto in_n = in[n];
        auto in_nm2 = in[n - 2];

        const double coeff_n = zernike_radon_coeff[n];
        const double coeff_nm2 = -zernike_radon_coeff[n - 2];
        auto out_n_flat = out_n.flatten();
        auto in_n_flat = in_n.flatten();
        auto in_nm2_flat = in_nm2.flatten();

        const std::size_t out_n_size = out_n_flat.size();
        const std::size_t in_n_size = in_n_flat.size();
        const std::size_t in_nm2_size = in_nm2_flat.size();
        [[assume(out_n_size == in_n_size)]];
        [[assume(out_n_size >= in_nm2_size)]];
        util::linear_combination(out_n_flat, coeff_n, in_n_flat, coeff_nm2, in_nm2_flat);

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = coeff_n*in_n_n[m, 0];
            out_n_n[m, 1] = coeff_n*in_n_n[m, 1];
        }
    }

    for (std::size_t n : out.indices(in.order()))
    {
        auto out_n = out[n];
        auto in_nm2 = in[n - 2];

        const double coeff_nm2 = -zernike_radon_coeff[n - 2];
        auto out_n_flat = out_n.flatten();
        auto in_nm2_flat = in_nm2.flatten();

        const std::size_t out_n_size = out_n_flat.size();
        const std::size_t in_nm2_size = in_nm2_flat.size();
        [[assume(out_n_size >= in_nm2_size)]];
        util::mul(out_n_flat, coeff_nm2, in_nm2_flat);

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = 0.0;
            out_n_n[m, 1] = 0.0;
        }
    }
}

void radon_transform(IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeSpan<const double>::shape_type::zernike_norm;

    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 <= out.order());

    if (in.order() == 0) return;

    const double coeff_0 = util::zernike_radon_coeff<zernike_norm>(0);
    out[0] = coeff_0*in[0];

    if (in.order() < 2)
    {
        out[2] = -coeff_0*in[0];
        return;
    }

    for (std::size_t n : in.indices(2))
    {
        const double coeff_n = util::zernike_radon_coeff<zernike_norm>(n);
        const double coeff_nm2 = util::zernike_radon_coeff<zernike_norm>(n - 2);
        out[n] = coeff_n*in[n] - coeff_nm2*in[n - 2];
    }

    const std::size_t nmax = util::even_floor(in.order() + 1);
    const double coeff_nm2 = util::zernike_radon_coeff<zernike_norm>(nmax - 2);
    out[nmax] = -coeff_nm2*in[nmax - 2];
}

void radon_transform(
    IsotropicZernikeSpan<const double> in, IsotropicZernikeSpan<double> out,
    IsotropicZernikeSpan<const double> zernike_radon_coeff) noexcept
{
    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 <= out.order());
    assert(zernike_radon_coeff.size() >= in.order());

    if (in.order() == 0) return;

    const double coeff_0 = zernike_radon_coeff[0];
    out[0] = coeff_0*in[0];

    if (in.order() < 2)
    {
        out[2] = -coeff_0*in[0];
        return;
    }

    for (std::size_t n : in.indices(2))
    {
        const double coeff_n = zernike_radon_coeff[n];
        const double coeff_nm2 = zernike_radon_coeff[n - 2];
        out[n] = coeff_n*in[n] - coeff_nm2*in[n - 2];
    }

    const std::size_t nmax = util::even_floor(in.order() + 1);
    const double coeff_nm2 = zernike_radon_coeff[nmax - 2];
    out[nmax] = -coeff_nm2*in[nmax - 2];
}

void radon_transform_inplace(
    ZernikeSpan<double> exp) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeSpan<const double>::shape_type::zernike_norm;

    const std::size_t order = exp.order();

    if (order < 3) return;

    for (std::size_t n = order - 1; n > std::max(order - 3, 1UL); --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_nm2 = -util::zernike_radon_coeff<zernike_norm>(n - 2);
        auto exp_n_flat = exp_n.flatten();
        auto exp_nm2_flat = exp_nm2.flatten();

        const std::size_t exp_n_size = exp_n_flat.size();
        const std::size_t exp_nm2_size = exp_nm2_flat.size();
        [[assume(exp_n_size >= exp_nm2_size)]];
        util::mul(exp_n_flat, coeff_nm2, exp_nm2_flat);

        auto out_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = 0.0;
            out_n_n[m, 1] = 0.0;

        }
    }

    for (std::size_t n = std::max(order - 3, 1UL); n > 1; --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_n = util::zernike_radon_coeff<zernike_norm>(n);
        const double coeff_nm2 = -util::zernike_radon_coeff<zernike_norm>(n - 2);
        auto exp_n_flat = exp_n.flatten();
        auto exp_nm2_flat = exp_nm2.flatten();

        for (std::size_t i = 0; i < exp_nm2.size(); ++i)
            exp_n_flat[i] = coeff_n*exp_n_flat[i] + coeff_nm2*exp_nm2_flat[i];

        auto exp_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            exp_n_n[m, 0] = coeff_n*exp_n_n[m, 0];
            exp_n_n[m, 1] = coeff_n*exp_n_n[m, 1];
        }
    }

    const double coeff_1 = util::zernike_radon_coeff<zernike_norm>(1);
    exp[1, 1, 0, 0] = coeff_1*exp[1, 1, 0, 0];
    exp[1, 1, 0, 1] = coeff_1*exp[1, 1, 0, 1];
    exp[1, 1, 1, 0] = coeff_1*exp[1, 1, 1, 0];
    exp[1, 1, 1, 1] = coeff_1*exp[1, 1, 1, 1];

    const double coeff_0 = util::zernike_radon_coeff<zernike_norm>(0);
    exp[0, 0, 0, 0] = coeff_0*exp[0, 0, 0, 0];
    exp[0, 0, 0, 1] = coeff_0*exp[0, 0, 0, 1];
}

void radon_transform_inplace(
    ZernikeSpan<double> exp, std::span<const double> zernike_radon_coeff) noexcept
{
    assert(zernike_radon_coeff.size() >= exp.order());
    const std::size_t order = exp.order();

    if (order < 3) return;

    for (std::size_t n = order - 1; n > std::max(order - 3, 1UL); --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_nm2 = -zernike_radon_coeff[n - 2];
        auto exp_n_flat = exp_n.flatten();
        auto exp_nm2_flat = exp_nm2.flatten();

        const std::size_t exp_n_size = exp_n_flat.size();
        const std::size_t exp_nm2_size = exp_nm2_flat.size();
        [[assume(exp_n_size >= exp_nm2_size)]];
        util::mul(exp_n_flat, coeff_nm2, exp_nm2_flat);

        auto out_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m, 0] = 0.0;
            out_n_n[m, 1] = 0.0;

        }
    }

    for (std::size_t n = std::max(order - 3, 1UL); n > 1; --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_n = zernike_radon_coeff[n];
        const double coeff_nm2 = -zernike_radon_coeff[n - 2];
        auto exp_n_flat = exp_n.flatten();
        auto exp_nm2_flat = exp_nm2.flatten();

        for (std::size_t i = 0; i < exp_nm2.size(); ++i)
            exp_n_flat[i] = coeff_n*exp_n_flat[i] + coeff_nm2*exp_nm2_flat[i];

        auto exp_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            exp_n_n[m, 0] = coeff_n*exp_n_n[m, 0];
            exp_n_n[m, 1] = coeff_n*exp_n_n[m, 1];
        }
    }

    const double coeff_1 = zernike_radon_coeff[1];
    exp[1, 1, 0, 0] = coeff_1*exp[1, 1, 0, 0];
    exp[1, 1, 0, 1] = coeff_1*exp[1, 1, 0, 1];
    exp[1, 1, 1, 0] = coeff_1*exp[1, 1, 1, 0];
    exp[1, 1, 1, 1] = coeff_1*exp[1, 1, 1, 1];

    const double coeff_0 = zernike_radon_coeff[0];
    exp[0, 0, 0, 0] = coeff_0*exp[0, 0, 0, 0];
    exp[0, 0, 0, 1] = coeff_0*exp[0, 0, 0, 1];
}

void radon_transform_inplace(
    IsotropicZernikeSpan<double> exp) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeSpan<const double>::shape_type::zernike_norm;

    const std::size_t order = exp.order();

    if (order < 3) return;

    const std::size_t nmax = util::even_floor(exp.order() - 1);
    const double coeff_nm2 = util::zernike_radon_coeff<zernike_norm>(nmax - 2);
    exp[nmax] = -coeff_nm2*exp[nmax - 2];

    for (std::size_t n = nmax - 2; n > 1; n -= 2)
    {
        const double coeff_n = util::zernike_radon_coeff<zernike_norm>(n);
        const double coeff_nm2 = util::zernike_radon_coeff<zernike_norm>(n - 2);
        exp[n] = coeff_n*exp[n] - coeff_nm2*exp[n - 2];
    }

    const double coeff_0 = util::zernike_radon_coeff<zernike_norm>(0);
    exp[0] = coeff_0*exp[0];
}

void radon_transform_inplace(
    IsotropicZernikeSpan<double> exp, IsotropicZernikeSpan<const double> zernike_radon_coeff) noexcept
{
    assert(zernike_radon_coeff.size() >= exp.order());
    const std::size_t order = exp.order();

    if (order < 3) return;

    const std::size_t nmax = util::even_floor(exp.order() - 1);
    const double coeff_nm2 = zernike_radon_coeff[nmax - 2];
    exp[nmax] = -coeff_nm2*exp[nmax - 2];

    for (std::size_t n = nmax - 2; n > 1; n -= 2)
    {
        const double coeff_n = zernike_radon_coeff[n];
        const double coeff_nm2 = zernike_radon_coeff[n - 2];
        exp[n] = coeff_n*exp[n] - coeff_nm2*exp[n - 2];
    }

    const double coeff_0 = zernike_radon_coeff[0];
    exp[0] = coeff_0*exp[0];
}

} // namespace zdm::zebra
