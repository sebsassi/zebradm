#include "radon_util.hpp"

namespace detail
{

void apply_legendre_shift(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> expansion, 
    zest::TriangleSpan<const double, zest::TriangleLayout> shifted_leg_coeff, 
    SHExpansionCollectionSpan<std::array<double, 2>> out)
{
    std::ranges::fill(out.flatten(), std::array<double, 2>{});
    const std::size_t order = expansion.order();
    for (std::size_t n = 0; n < order; ++n)
    {
        auto zernike_n = expansion[n];
        auto coeff_n = shifted_leg_coeff[n];
        for (std::size_t lp = 0; lp <= n; ++lp)
        {
            const double coeff_n_lp = coeff_n[lp];
            auto out_lp = out[lp];
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                auto zernike_n_l = zernike_n[l];
                auto out_lp_l = out_lp[l];
                for (std::size_t m = 0; m <= l; ++m)
                    out_lp_l[m] += coeff_n_lp*zernike_n_l[m];
            }
        }
    }
}

void apply_gegenbauer_recursion(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    constexpr zest::zt::ZernikeNorm NORM = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>::zernike_norm;

    assert(in.order() + 2 == out.order());
    const std::size_t out_order = out.order();
    const std::size_t in_order = in.order();

    if (in_order == 0) return;
    
    out(0,0,0) = geg_rec_coeff<NORM>(0)*in(0,0,0);

    if (in_order > 1)
    {
        constexpr double coeff = geg_rec_coeff<NORM>(1);
        out(1,1,0) = coeff*in(1,1,0);
        out(1,1,1) = coeff*in(1,1,1);
    }
    else
    {
        out(2,0,0) = (-geg_rec_coeff<NORM>(0))*in(0,0,0);
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

        const double coeff_n = geg_rec_coeff<NORM>(n);
        const double coeff_nm2 = -geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_n_l = in_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
                out_n_l[m] = coeff_n*in_n_l[m] + coeff_nm2*in_nm2_l[m];
        }

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = coeff_n*in_n_n[m];
    }

    for (std::size_t n = std::max(in_order, 2UL); n < out_order; ++n)
    {
        auto out_n = out[n];
        auto in_nm2 = in[n - 2];

        const double coeff_nm2 = -geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
                out_n_l[m] = coeff_nm2*coeff_nm2*in_nm2_l[m];
        }

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }
}

}