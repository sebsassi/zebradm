#include "radon_util.hpp"

namespace util
{

/**
    @brief Apply the transformation `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)` to a Zernike expansion.

    @param in input Zernike expansion `f_{nlm}`
    @param out output Zernike expansion `g_{nlm}`

    @note This transformation appears in the evaluation of the Zernike based Radon transform, where it reduces an expression `f_{nlm}(1 - x^2)C^{3/2}_n(x)` to the form `g_{nlm}P_n(x)` with `g_{nlm}` given as above.
*/
void apply_gegenbauer_reduction(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out) noexcept
{
    constexpr zest::zt::ZernikeNorm NORM = zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>>::zernike_norm;

    assert(in.order() + 2 == out.order());
    const std::size_t out_order = out.order();
    const std::size_t in_order = in.order();

    if (in_order == 0) return;
    
    out(0,0,0) = {
        geg_rec_coeff<NORM>(0)*in(0,0,0)[0],
        geg_rec_coeff<NORM>(0)*in(0,0,0)[1]
    };

    if (in_order > 1)
    {
        const double coeff = geg_rec_coeff<NORM>(1);
        out(1,1,0) = {coeff*in(1,1,0)[0], coeff*in(1,1,0)[1]};
        out(1,1,1) = {coeff*in(1,1,1)[0], coeff*in(1,1,1)[1]};
    }
    else
    {
        out(2,0,0) = {(-geg_rec_coeff<NORM>(0))*in(0,0,0)[0], (-geg_rec_coeff<NORM>(0))*in(0,0,0)[1]};
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

        const double coeff_nm2 = -geg_rec_coeff<NORM>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                out_n_l[m][0] = coeff_nm2*coeff_nm2*in_nm2_l[m][0];
                out_n_l[m][1] = coeff_nm2*coeff_nm2*in_nm2_l[m][1];
            }
        }

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }
}

}