#include "zebra_radon.hpp"

#include <cassert>

#include "radon_util.hpp"

namespace zebra
{

void radon_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out) noexcept
{
    constexpr zest::zt::ZernikeNorm NORM = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>::zernike_norm;

    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 == out.order());
    const std::size_t out_order = out.order();
    const std::size_t in_order = in.order();

    if (in_order == 0) return;
    
    out(0,0,0) = {
        util::geg_rec_coeff<NORM>(0)*in(0,0,0)[0],
        util::geg_rec_coeff<NORM>(0)*in(0,0,0)[1]
    };

    if (in_order > 1)
    {
        const double coeff = util::geg_rec_coeff<NORM>(1);
        out(1,1,0) = {coeff*in(1,1,0)[0], coeff*in(1,1,0)[1]};
        out(1,1,1) = {coeff*in(1,1,1)[0], coeff*in(1,1,1)[1]};
    }
    else
    {
        out(2,0,0) = {(-util::geg_rec_coeff<NORM>(0))*in(0,0,0)[0], (-util::geg_rec_coeff<NORM>(0))*in(0,0,0)[1]};
        out(2,2,0) = std::array<double, 2>{};
        out(2,2,1) = std::array<double, 2>{};
        out(2,2,2) = std::array<double, 2>{};
        return;
    }

    for (std::size_t n = 2; n < in_order; ++n)
    {
        auto out_n = out[n];
        auto in_n = in[n];
        auto in_nm2 = in[n - 2];

        const double coeff_n = util::geg_rec_coeff<NORM>(n);
        const double coeff_nm2 = -util::geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_n_l = in_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                out_n_l[m][0] = coeff_n*in_n_l[m][0] + coeff_nm2*in_nm2_l[m][0];
                out_n_l[m][1] = coeff_n*in_n_l[m][1] + coeff_nm2*in_nm2_l[m][1];
            }
        }

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m][0] = coeff_n*in_n_n[m][0];
            out_n_n[m][1] = coeff_n*in_n_n[m][1];
        }
    }

    for (std::size_t n = std::max(in_order, 2UL); n < out_order; ++n)
    {
        auto out_n = out[n];
        auto in_nm2 = in[n - 2];

        const double coeff_nm2 = -util::geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                out_n_l[m][0] = coeff_nm2*in_nm2_l[m][0];
                out_n_l[m][1] = coeff_nm2*in_nm2_l[m][1];
            }
        }

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }
}

void radon_transform_inplace(
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> exp) noexcept
{
    constexpr zest::zt::ZernikeNorm NORM = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>::zernike_norm;

    const std::size_t order = exp.order();

    if (order < 3) return;

    for (std::size_t n = order - 1; n > std::max(order - 3, 1UL); --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_nm2 = -util::geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto exp_n_l = exp_n[l];
            auto exp_nm2_l = exp_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                exp_n_l[m][0] = coeff_nm2*exp_nm2_l[m][0];
                exp_n_l[m][1] = coeff_nm2*exp_nm2_l[m][1];
            }
        }

        auto out_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }

    for (std::size_t n = std::max(order - 3, 1UL); n > 1; --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_n = util::geg_rec_coeff<NORM>(n);
        const double coeff_nm2 = -util::geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto exp_n_l = exp_n[l];
            auto exp_nm2_l = exp_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                exp_n_l[m][0] = coeff_n*exp_n_l[m][0] + coeff_nm2*exp_nm2_l[m][0];
                exp_n_l[m][1] = coeff_n*exp_n_l[m][1] + coeff_nm2*exp_nm2_l[m][1];
            }
        }

        auto exp_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            exp_n_n[m][0] = coeff_n*exp_n_n[m][0];
            exp_n_n[m][1] = coeff_n*exp_n_n[m][1];
        }
    }

    const double coeff = util::geg_rec_coeff<NORM>(1);
    exp(1,1,0) = {coeff*exp(1,1,0)[0], coeff*exp(1,1,0)[1]};
    exp(1,1,1) = {coeff*exp(1,1,1)[0], coeff*exp(1,1,1)[1]};

    exp(0,0,0) = {
        util::geg_rec_coeff<NORM>(0)*exp(0,0,0)[0],
        util::geg_rec_coeff<NORM>(0)*exp(0,0,0)[1]
    };
}

}