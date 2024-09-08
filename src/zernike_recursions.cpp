#include "zernike_recursions.hpp"

#include <algorithm>
#include <cassert>

namespace zebra
{
namespace detail
{

ZernikeRecursionData::ZernikeRecursionData(std::size_t order):
    m_sqrt_n(2*order + 1), m_inv_sqrt_2np1_2np3(order + 2), m_order(order)
{
    if (order == 0) return;

    for (std::size_t n = 1; n < m_sqrt_n.size(); ++n)
        m_sqrt_n[n] = std::sqrt(double(n));
    
    for (std::size_t n = 0; n < m_inv_sqrt_2np1_2np3.size(); ++n)
    {
        const double _2n = double(2*n);
        m_inv_sqrt_2np1_2np3[n] = std::sqrt((_2n + 1.0)*(_2n + 3.0));
    }
}

void ZernikeRecursionData::expand(std::size_t order)
{
    if (order <= m_order) return;

    const std::size_t old_n_size = m_sqrt_n.size();
    const std::size_t old_2np1_2np3_size = m_inv_sqrt_2np1_2np3.size();

    const std::size_t new_n_size = 2*order + 1;
    const std::size_t new_2np1_2np3_size = order + 2;

    m_sqrt_n.resize(new_n_size);
    m_inv_sqrt_2np1_2np3.resize(new_2np1_2np3_size);

    for (std::size_t n = old_n_size; n < new_n_size; ++n)
        m_sqrt_n[n] = std::sqrt(double(n));
    
    for (std::size_t n = old_2np1_2np3_size; n < new_2np1_2np3_size; ++n)
    {
        const double _2n = double(2*n);
        m_inv_sqrt_2np1_2np3[n] = std::sqrt((_2n + 1.0)*(_2n + 3.0));
    }
}

enum class PlaneCoord { X, Y };

template <PlaneCoord COORD>
void multiply_by_x_y_impl(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    /*
    Literally the worst function I have ever written.

    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 1, l + 1, m + 1)
    in(n + 1, l + 1, m - 1)
    in(n + 1, l - 1, m + 1)
    in(n + 1, l - 1, m - 1)
    in(n - 1, l + 1, m + 1)
    in(n - 1, l + 1, m - 1)
    in(n - 1, l - 1, m + 1)
    in(n - 1, l - 1, m - 1)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */

    constexpr double sqrt2 = std::numbers::sqrt2;
    constexpr double sqrt3 = std::numbers::sqrt3;
    constexpr double sqrt5 = 2.2360679774997896964091737;

    assert(in.order() < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order();
    if (nmax == 0) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});

    if constexpr (COORD == PlaneCoord::X)
    {
        out(1,1,1)[0] = (-1.0/sqrt5)*in(0,0,0)[0];
        if (nmax == 1) return;

        out(2,0,0)[0] = (-1.0/(sqrt2*sqrt3*sqrt5*sqrt7))*in(1,1,1)[0];
        out(2,2,0)[0] = (-1.0/(6.0*sqrt2*sqrt5))*in(1,1,1)[0];
        out(2,2,1)[0] = (2.0/(sqrt3*sqrt5))*in(1,1,0)[0];
        out(2,2,2)[0] = (-2.0/sqrt7)*in(1,1,1)[0];
        out(2,2,2)[1] = (-2.0/sqrt7)*in(1,1,1)[1];
        if (nmax == 2) return;
        
        out(0,0,0)[0] = (-0.5/(sqrt2*sqrt5))*in(1,1,1)[0];

        if (nmax > 2)
        {
            out(1,1,0)[0] = (-0.5/(sqrt2*sqrt3*sqrt7))*in(2,2,1)[0];
        
            out(1,1,1)[0] -= (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[0]
                    - (1.0/(sqrt3*sqrt7))*in(2,2,0)[0]
                    + (0.5/(sqrt3*sqrt7))*in(2,2,2)[0];

            out(1,1,1)[1] -= (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[1]
                    - (1.0/(sqrt3*sqrt7))*in(2,2,0)[1]
                    + (0.5/(sqrt3*sqrt7))*in(2,2,2)[1];
            
            /* TODO n = 2 */
            out(2,0,0)[0] += in(3,1,1)[0];
            out(2,2,0)[0] += in(3,1,1)[0] + in(3,3,1)[0];
            out(2,2,1)[0] += in(3,1,0)[0] + in(3,3,0)[0] + in(3,3,2)[0];
            out(2,2,1)[1] += in(3,3,2)[1];
            out(2,2,2)[0] += in(3,1,1)[0] + in(3,3,1)[0] + in(3,3,3)[0];
            out(2,2,2)[1] += in(3,1,1)[1] + in(3,3,1)[1] + in(3,3,3)[1];
        }
    }
    else
    {
        out(1,1,1)[1] = (1.0/sqrt5)*in(0,0,0)[0];
        if (nmax == 1) return;

        out(2,0,0)[0] = (1.0/(sqrt2*sqrt3*sqrt5*sqrt7))*in(1,1,1)[1];
        out(2,2,0)[0] = (-1.0/(6.0*sqrt2*sqrt5))*in(1,1,1)[1];
        out(2,2,1)[1] = (2.0/(sqrt3*sqrt5))*in(1,1,0)[0];
        out(2,2,2)[0] = (2.0/sqrt7)*in(1,1,1)[1];
        out(2,2,2)[1] = (-2.0/sqrt7)*in(1,1,1)[0];
        if (nmax == 2) return;

        out(0,0,0)[0] = (0.5/(sqrt2*sqrt5))*in(1,1,1)[1];

        if (nmax > 2)
        {
            out(1,1,0)[0] = (0.5/(sqrt2*sqrt3*sqrt7))*in(2,2,1)[1];
        
            out(1,1,1)[0] -= (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[1]
                    - (1.0/(sqrt3*sqrt7))*in(2,2,0)[1]
                    - (0.5/(sqrt3*sqrt7))*in(2,2,2)[1];

            out(1,1,1)[1] += (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[0]
                    - (1.0/(sqrt3*sqrt7))*in(2,2,0)[0]
                    - (0.5/(sqrt3*sqrt7))*in(2,2,2)[0];

            /* TODO n = 2 */
            out(2,0,0)[0] += in(3,1,1)[0];
            out(2,2,0)[0] += in(3,1,1)[0] + in(3,3,1)[0];
            out(2,2,1)[0] += in(3,3,2)[1];
            out(2,2,1)[1] += in(3,1,0)[0] + in(3,3,0)[0] + in(3,3,2)[0];
            out(2,2,2)[0] += in(3,1,1)[1] + in(3,3,1)[1] + in(3,3,3)[1];
            out(2,2,2)[1] += in(3,1,1)[0] + in(3,3,1)[0] + in(3,3,3)[0];
        }
    }

    for (std::size_t n = 3; n < nmax; ++n)
    {
        const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_p = 0.5*coeff_data.inv_sqrt_2np3_2np5(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        auto in_np1 = in[n + 1];

        const std::size_t n_parity = n & 1;
        
        // edge case: n > 0, l = 0, m = 0
        if (n_parity == 0)
        {
            constexpr double l_coeff = (COORD == PlaneCoord::X) ?
                -1.0/(sqrt2*sqrt3) : 1.0/(sqrt2*sqrt3);
            const double dn = double(n);
            out_n(0,0)[0] = l_coeff*(
                    n_coeff_m*dn*in_nm1(1,1)[0]
                    + n_coeff_p*(dn + 3.0)*in_np1(1,1)[0]);
            
            // l == 2 special case
            constexpr double l_coeff_m = 1.0/(sqrt3*sqrt5);
            constexpr double l_coeff_p = 1.0/(sqrt5*sqrt7);

            // l == 2 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 3.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 2.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*dn;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 5.0);
            
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n(2,0)[0] = (1.0/sqrt2)*(
                      nl_coeff_mm*in_nm1(1,1)[0] - nl_coeff_mp*in_nm1(3,1)[0]
                    + nl_coeff_pm*in_np1(1,1)[0] - nl_coeff_pp*in_np1(3,1)[0]);
                
                constexpr double l1_coeff = 2.0*sqrt3;
                out_n(2,1)[0]
                    = -nl_coeff_mm*l1_coeff*in_nm1(1,0)[0]
                    - nl_coeff_mp*(in_nm1(3,2)[0] - l1_coeff*in_nm1(3,0)[0])
                    - nl_coeff_pm*l1_coeff*in_np1(1,0)[0]
                    - nl_coeff_pp*(in_np1(3,2)[0] - l1_coeff*in_np1(3,0)[0]);
                
                out_n(2,1)[1]
                    = -nl_coeff_mp*in_nm1(3,2)[1] - nl_coeff_pp*in_np1(3,2)[1];
                    
                constexpr double l2_coeff_m = 2.0*sqrt3;
                constexpr double l2_coeff_p = sqrt2;
                out_n(2,2)[0]
                    = -nl_coeff_mm*l2_coeff_m*in_nm1(1,1)[0]
                    - nl_coeff_mp*(in_nm1(3,3)[0] - l2_coeff_p*in_nm1(3,1)[0])
                    - nl_coeff_pm*l2_coeff_m*in_np1(1,1)[0]
                    - nl_coeff_pp*(in_np1(3,3)[0] - l2_coeff_p*in_np1(3,1)[0]);
                
                out_n(2,2)[1]
                    = -nl_coeff_mm*l2_coeff_m*in_nm1(1,1)[1]
                    - nl_coeff_mp*(in_nm1(3,3)[1] - l2_coeff_p*in_nm1(3,1)[1])
                    - nl_coeff_pm*l2_coeff_m*in_np1(1,1)[1]
                    - nl_coeff_pp*(in_np1(3,3)[1] - l2_coeff_p*in_np1(3,1)[1]);
            }
            else
            {
                out_n(2,0)[0] = (1.0/sqrt2)*(
                      nl_coeff_pp*in_np1(3,1)[0] + nl_coeff_mp*in_nm1(3,1)[0]
                    - nl_coeff_pm*in_np1(1,1)[0] - nl_coeff_mm*in_nm1(1,1)[0]);
                
                constexpr double l1_coeff = 2.0*sqrt3;
                out_n(2,1)[0]
                    = nl_coeff_pp*in_np1(3,2)[1] + nl_coeff_mp*in_nm1(3,2)[1];
                
                out_n(2,1)[1]
                    = +nl_coeff_mm*l1_coeff*in_nm1(1,0)[0]
                    - nl_coeff_mp*(in_nm1(3,2)[0] + l1_coeff*in_nm1(3,0)[0])
                    + nl_coeff_pm*l1_coeff*in_np1(1,0)[0]
                    - nl_coeff_pp*(in_np1(3,2)[0] + l1_coeff*in_np1(3,0)[0]);
                    
                constexpr double l2_coeff_m = 2.0*sqrt3;
                constexpr double l2_coeff_p = sqrt2;
                out_n(2,2)[0]
                    = nl_coeff_pp*(in_np1(3,3)[1] + l2_coeff_p*in_np1(3,1)[1])
                    + nl_coeff_mp*(in_nm1(3,3)[1] + l2_coeff_p*in_nm1(3,1)[1])
                    - nl_coeff_pm*l2_coeff_m*in_np1(1,1)[1]
                    - nl_coeff_mm*l2_coeff_m*in_nm1(1,1)[1];
                
                out_n(2,2)[1]
                    = nl_coeff_mm*l2_coeff_m*in_nm1(1,1)[0]
                    + nl_coeff_pm*l2_coeff_m*in_np1(1,1)[0]
                    - nl_coeff_mp*(in_nm1(3,3)[0] + l2_coeff_p*in_nm1(3,1)[0])
                    - nl_coeff_pp*(in_np1(3,3)[0] + l2_coeff_p*in_np1(3,1)[0]);
            }
        }
        else
        {
            // l == 1 special case
            constexpr double l_coeff_m = 1.0/(sqrt3);
            constexpr double l_coeff_p = 1.0/(sqrt3*sqrt5);

            // l == 1 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 2.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 1.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*(dn + 1.0);
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 4.0);

            if constexpr (COORD == PlaneCoord::X)
            {
                out_n(1,0)[0] = (-1.0/sqrt2)*(
                    nl_coeff_mp*in_nm1(2,1)[0] + nl_coeff_pp*in_np1(2,1)[0]);

                constexpr double l1_coeff = 2.0;
                out_n(1,1)[0]
                    = -nl_coeff_mm*l1_coeff*in_nm1(0,0)[0]
                    - nl_coeff_mp*(in_nm1(2,2)[0] - l1_coeff*in_nm1(2,0)[0])
                    - nl_coeff_pm*l1_coeff*in_np1(0,0)[0]
                    - nl_coeff_pp*(in_np1(2,2)[0] - l1_coeff*in_np1(2,0)[0]);

                out_n(1,1)[1]
                    = -nl_coeff_mp*in_nm1(2,2)[1] - nl_coeff_pp*in_np1(2,2)[1];
            }
            else
            {
                out_n(1,0)[0] = (1.0/sqrt2)*(
                    nl_coeff_mp*in_nm1(2,1)[1] + nl_coeff_pp*in_np1(2,1)[1]);

                constexpr double l1_coeff = 2.0;
                out_n(1,1)[0]
                    = nl_coeff_mp*in_nm1(2,2)[1] + nl_coeff_pp*in_np1(2,2)[1];
                
                out_n(1,1)[1]
                    = nl_coeff_mm*l1_coeff*in_nm1(0,0)[0]
                    - nl_coeff_mp*(in_nm1(2,2)[0] - l1_coeff*in_nm1(2,0)[0])
                    + nl_coeff_pm*l1_coeff*in_np1(0,0)[0]
                    - nl_coeff_pp*(in_np1(2,2)[0] - l1_coeff*in_np1(2,0)[0]);
            }
        }

        for (std::size_t l = 4 - n_parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double nml2 = nml + 2.0;
            const double npl3 = npl + 3.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*nml2;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];
            auto in_np1_lm1 = in_np1[l - 1];
            auto in_np1_lp1 = in_np1[l + 1];

            const double l1_coeff_mmp = coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
            const double l1_coeff_pmp = coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
            const double l1_coeff_mmm = coeff_data.sqrt_n(l - 2)*coeff_data.sqrt_n(l - 1);
            const double l1_coeff_pmm = coeff_data.sqrt_n(l + 2)*coeff_data.sqrt_n(l + 3);

            // edge case: n > 0, l > 0, m = 0
            // edge case: n > 0, l > 0, m = 1
            /* TODO */
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n_l[0][0] = (1.0/sqrt2)*(
                        nl_coeff_mm*in_nm1_lm1[1][0]
                        - nl_coeff_mp*in_nm1_lp1[1][0]
                        + nl_coeff_pm*in_np1_lm1[1][0]
                        - nl_coeff_pp*in_np1_lp1[1][0]);
                
                out_n_l[1][0]
                    = nl_coeff_mm*(
                        in_nm1_lm1[2][0] - l1_coeff_mmp*sqrt2*in_nm1_lm1[0][0])
                    - nl_coeff_mp*(
                        in_nm1_lp1[2][0] - l1_coeff_pmp*sqrt2*in_nm1_lp1[0][0])
                    + nl_coeff_pm*(
                        in_np1_lm1[2][0] - l1_coeff_mmp*sqrt2*in_np1_lm1[0][0])
                    - nl_coeff_pp*(
                        in_np1_lp1[2][0] - l1_coeff_pmp*sqrt2*in_np1_lp1[0][0]);
                
                out_n_l[1][1]
                    = nl_coeff_mm*(
                        sqrt2*in_nm1_lm1[0][1] - l1_coeff_mmm*in_nm1_lm1[2][1])
                    - nl_coeff_mp*(
                        sqrt2*in_nm1_lp1[0][1] - l1_coeff_pmm*in_nm1_lp1[2][1])
                    + nl_coeff_pm*(
                        sqrt2*in_np1_lm1[0][1] - l1_coeff_mmm*in_np1_lm1[2][1])
                    - nl_coeff_pp*(
                        sqrt2*in_np1_lp1[0][1] - l1_coeff_pmm*in_np1_lp1[2][1]);
            }
            else
            {
                out_n_l[0][0] = (1.0/sqrt2)*(
                        nl_coeff_pp*in_np1_lp1[1][0]
                        + nl_coeff_mp*in_nm1_lp1[1][0]
                        - nl_coeff_pm*in_np1_lm1[1][0]
                        - nl_coeff_mm*in_nm1_lm1[1][0]);
                
                out_n_l[1][0]
                    = nl_coeff_pp*(
                        in_np1_lp1[2][0] + l1_coeff_pmp*in_np1_lp1[0][0])
                    + nl_coeff_mp*(
                        in_nm1_lp1[2][0] + l1_coeff_pmp*in_nm1_lp1[0][0])
                    - nl_coeff_pm*(
                        in_np1_lm1[2][0] + l1_coeff_mmp*in_np1_lm1[0][0])
                    - nl_coeff_mm*(
                        in_nm1_lm1[2][0] + l1_coeff_mmp*in_nm1_lm1[0][0]);
                
                out_n_l[1][1]
                    = nl_coeff_mm*(
                        in_nm1_lm1[0][1] + l1_coeff_mmm*in_nm1_lm1[2][1])
                    - nl_coeff_mp*(
                        in_nm1_lp1[0][1] + l1_coeff_pmm*in_nm1_lp1[2][1])
                    + nl_coeff_pm*(
                        in_np1_lm1[0][1] + l1_coeff_mmm*in_np1_lm1[2][1])
                    - nl_coeff_pp*(
                        in_np1_lp1[0][1] + l1_coeff_pmm*in_np1_lp1[2][1]);
            }

            for (std::size_t m = 2; m < l - 1; ++m)
            {
                const double lm_coeff_mmp = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_pmp = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_mmm = coeff_data.sqrt_n(l - m - 1)*coeff_data.sqrt_n(l - m);
                const double lm_coeff_pmm = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lm1_mp1 = in_nm1_lm1[m + 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];
                const std::array<double, 2> in_np1_lm1_mm1 = in_np1_lm1[m - 1];
                const std::array<double, 2> in_np1_lm1_mp1 = in_np1_lm1[m + 1];
                const std::array<double, 2> in_np1_lp1_mm1 = in_np1_lp1[m - 1];
                const std::array<double, 2> in_np1_lp1_mp1 = in_np1_lp1[m + 1];

                // base case
                /* TODO */
                if constexpr (COORD == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*(
                            in_nm1_lm1_mp1[0] - lm_coeff_mmp*in_nm1_lm1_mm1[0])
                        - nl_coeff_mp*(
                            in_nm1_lp1_mp1[0] - lm_coeff_pmp*in_nm1_lp1_mm1[0])
                        + nl_coeff_pm*(
                            in_np1_lm1_mp1[0] - lm_coeff_mmp*in_np1_lm1_mm1[0])
                        - nl_coeff_pp*(
                            in_np1_lp1_mp1[0] - lm_coeff_pmp*in_np1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*(
                            in_nm1_lm1_mm1[1] - lm_coeff_mmm*in_nm1_lm1_mp1[1])
                        - nl_coeff_mp*(
                            in_nm1_lp1_mm1[1] - lm_coeff_pmm*in_nm1_lp1_mp1[1])
                        + nl_coeff_pm*(
                            in_np1_lm1_mm1[1] - lm_coeff_mmm*in_np1_lm1_mp1[1])
                        - nl_coeff_pp*(
                            in_np1_lp1_mm1[1] - lm_coeff_pmm*in_np1_lp1_mp1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        = nl_coeff_pp*(
                            in_np1_lp1_mp1[0] + lm_coeff_pmp*in_np1_lp1_mm1[0])
                        + nl_coeff_mp*(
                            in_nm1_lp1_mp1[0] + lm_coeff_pmp*in_nm1_lp1_mm1[0])
                        - nl_coeff_pm*(
                            in_np1_lm1_mp1[0] + lm_coeff_mmp*in_np1_lm1_mm1[0])
                        - nl_coeff_mm*(
                            in_nm1_lm1_mp1[0] + lm_coeff_mmp*in_nm1_lm1_mm1[0]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*(
                            in_nm1_lm1_mm1[1] + lm_coeff_mmm*in_nm1_lm1_mp1[1])
                        - nl_coeff_mp*(
                            in_nm1_lp1_mm1[1] + lm_coeff_pmm*in_nm1_lp1_mp1[1])
                        + nl_coeff_pm*(
                            in_np1_lm1_mm1[1] + lm_coeff_mmm*in_np1_lm1_mp1[1])
                        - nl_coeff_pp*(
                            in_np1_lp1_mm1[1] + lm_coeff_pmm*in_np1_lp1_mp1[1]);
                }
            }

            // edge case: n > 0, l > 2, m = l - 1
            // edge case: n > 0, l > 1, m = l
            for (std::size_t m = std::max(l - 1, 2UL); m <= l; ++m)
            {
                const double lm_coeff_mmp = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_pmp = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_pmm = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];
                const std::array<double, 2> in_np1_lm1_mm1 = in_np1_lm1[m - 1];
                const std::array<double, 2> in_np1_lp1_mm1 = in_np1_lp1[m - 1];
                const std::array<double, 2> in_np1_lp1_mp1 = in_np1_lp1[m + 1];

                /* TODO */
                if constexpr (COORD == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = -nl_coeff_mm*lm_coeff_mmp*in_nm1_lm1_mm1[0]
                        - nl_coeff_mp*(
                            in_nm1_lp1_mp1[0] - lm_coeff_pmp*in_nm1_lp1_mm1[0])
                        - nl_coeff_pm*lm_coeff_mmp*in_np1_lm1_mm1[0]
                        - nl_coeff_pp*(
                            in_np1_lp1_mp1[0] - lm_coeff_pmp*in_np1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            in_nm1_lp1_mm1[1] - lm_coeff_pmm*in_nm1_lp1_mp1[1])
                        + nl_coeff_pm*in_np1_lm1_mm1[1]
                        - nl_coeff_pp*(
                            in_np1_lp1_mm1[1] - lm_coeff_pmm*in_np1_lp1_mp1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        =  nl_coeff_pp*(
                            in_np1_lp1_mp1[0] + lm_coeff_pmp*in_np1_lp1_mm1[0])
                        + nl_coeff_mp*(
                            in_nm1_lp1_mp1[0] + lm_coeff_pmp*in_nm1_lp1_mm1[0])
                        - nl_coeff_pm*lm_coeff_mmp*in_np1_lm1_mm1[0]
                        - nl_coeff_mm*lm_coeff_mmp*in_nm1_lm1_mm1[0];
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            in_nm1_lp1_mm1[1] + lm_coeff_pmm*in_nm1_lp1_mp1[1])
                        + nl_coeff_pm*in_np1_lm1_mm1[1]
                        - nl_coeff_pp*(
                            in_np1_lp1_mm1[1] + lm_coeff_pmm*in_np1_lp1_mp1[1]);
                }
            }
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(n);
        const double npl = double(2*n);
        const double npl1 = npl + 1.0;
        const double npl3 = npl + 3.0;

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_pm = n_coeff_p*l_coeff_m*2.0;
        const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];
        auto in_np1_nm1 = in_np1[n - 1];
        auto in_np1_np1 = in_np1[n + 1];

        const double n1_coeff_mmp = coeff_data.sqrt_n(n)*coeff_data.sqrt_n(n + 1);
        const double n1_coeff_pmp = coeff_data.sqrt_n(n)*coeff_data.sqrt_n(n + 1);
        const double n1_coeff_mmm = coeff_data.sqrt_n(n - 2)*coeff_data.sqrt_n(n - 1);
        const double n1_coeff_pmm = coeff_data.sqrt_n(n + 2)*coeff_data.sqrt_n(n + 3);

        // edge case: n > 0, l = n, m = 0
        // edge case: n > 0, l = n, m = 1
        /* TODO */
        if constexpr (COORD == PlaneCoord::X)
        {
            out_n_n[0][0] = (1.0/sqrt2)*(
                    nl_coeff_mm*in_nm1_nm1[1][0]
                    + nl_coeff_pm*in_np1_nm1[1][0]
                    - nl_coeff_pp*in_np1_np1[1][0]);
            
            out_n_n[1][0]
                = nl_coeff_mm*(
                    in_nm1_nm1[2][0] - n1_coeff_mmp*sqrt2*in_nm1_nm1[0][0])
                + nl_coeff_pm*(
                    in_np1_nm1[2][0] - n1_coeff_mmp*sqrt2*in_np1_nm1[0][0])
                - nl_coeff_pp*(
                    in_np1_np1[2][0] - n1_coeff_pmp*sqrt2*in_np1_np1[0][0]);
            
            out_n_n[1][1]
                = nl_coeff_mm*(
                    sqrt2*in_nm1_nm1[0][1] - n1_coeff_mmm*in_nm1_nm1[2][1])
                + nl_coeff_pm*(
                    sqrt2*in_np1_nm1[0][1] - n1_coeff_mmm*in_np1_nm1[2][1])
                - nl_coeff_pp*(
                    sqrt2*in_np1_np1[0][1] - n1_coeff_pmm*in_np1_np1[2][1]);
        }
        else
        {
            out_n_n[0][0] = (1.0/sqrt2)*(
                    nl_coeff_pp*in_np1_np1[1][0]
                    - nl_coeff_pm*in_np1_nm1[1][0]
                    - nl_coeff_mm*in_nm1_nm1[1][0]);
            
            out_n_n[1][0]
                = nl_coeff_pp*(
                    in_np1_np1[2][0] + n1_coeff_pmp*in_np1_np1[0][0])
                - nl_coeff_pm*(
                    in_np1_nm1[2][0] + n1_coeff_mmp*in_np1_nm1[0][0])
                - nl_coeff_mm*(
                    in_nm1_nm1[2][0] + n1_coeff_mmp*in_nm1_nm1[0][0]);
            
            out_n_n[1][1]
                = nl_coeff_mm*(
                    in_nm1_nm1[0][1] + n1_coeff_mmm*in_nm1_nm1[2][1])
                + nl_coeff_pm*(
                    in_np1_nm1[0][1] + n1_coeff_mmm*in_np1_nm1[2][1])
                - nl_coeff_pp*(
                    in_np1_np1[0][1] + n1_coeff_pmm*in_np1_np1[2][1]);
        }
        
        for (std::size_t m = 2; m < n - 1; ++m)
        {
            const double lm_coeff_mmp = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_pmp = coeff_data.sqrt_n(n - m + 1)*coeff_data.sqrt_n(n - m + 2);
            const double lm_coeff_mmm = coeff_data.sqrt_n(n - m - 1)*coeff_data.sqrt_n(n - m);
            const double lm_coeff_pmm = coeff_data.sqrt_n(n + m + 1)*coeff_data.sqrt_n(n + m + 2);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
            const std::array<double, 2> in_nm1_nm1_mp1 = in_nm1_nm1[m + 1];
            const std::array<double, 2> in_np1_nm1_mm1 = in_np1_nm1[m - 1];
            const std::array<double, 2> in_np1_nm1_mp1 = in_np1_nm1[m + 1];
            const std::array<double, 2> in_np1_np1_mm1 = in_np1_np1[m - 1];
            const std::array<double, 2> in_np1_np1_mp1 = in_np1_np1[m + 1];

            // edge case: n > 0, l = n, m > 1
            /* TODO */
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n_n_m[0]
                    = nl_coeff_mm*(
                        in_nm1_nm1_mp1[0] - lm_coeff_mmp*in_nm1_nm1_mm1[0])
                    + nl_coeff_pm*(
                        in_np1_nm1_mp1[0] - lm_coeff_mmp*in_np1_nm1_mm1[0])
                    - nl_coeff_pp*(
                        in_np1_np1_mp1[0] - lm_coeff_pmp*in_np1_np1_mm1[0]);
                
                out_n_n_m[1]
                    = nl_coeff_mm*(
                        in_nm1_nm1_mm1[1] - lm_coeff_mmm*in_nm1_nm1_mp1[1])
                    + nl_coeff_pm*(
                        in_np1_nm1_mm1[1] - lm_coeff_mmm*in_np1_nm1_mp1[1])
                    - nl_coeff_pp*(
                        in_np1_np1_mm1[1] - lm_coeff_pmm*in_np1_np1_mp1[1]);
            }
            else
            {
                out_n_n_m[0]
                    = nl_coeff_pp*(
                        in_np1_np1_mp1[0] + lm_coeff_pmp*in_np1_np1_mm1[0])
                    - nl_coeff_pm*(
                        in_np1_nm1_mp1[0] + lm_coeff_mmp*in_np1_nm1_mm1[0])
                    - nl_coeff_mm*(
                        in_nm1_nm1_mp1[0] + lm_coeff_mmp*in_nm1_nm1_mm1[0]);
                
                out_n_n_m[1]
                    = nl_coeff_mm*(
                        in_nm1_nm1_mm1[1] + lm_coeff_mmm*in_nm1_nm1_mp1[1])
                    + nl_coeff_pm*(
                        in_np1_nm1_mm1[1] + lm_coeff_mmm*in_np1_nm1_mp1[1])
                    - nl_coeff_pp*(
                        in_np1_np1_mm1[1] + lm_coeff_pmm*in_np1_np1_mp1[1]);
            }
        }

        // edge case: n > 0, l = n, m = l - 1
        // edge case: n > 0, l = n, m = l
        for (std::size_t m = std::max(n - 1, 2UL); m <= n; ++m)
        {
            const double lm_coeff_mmp = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_pmp = coeff_data.sqrt_n(n - m + 1)*coeff_data.sqrt_n(n - m + 2);
            const double lm_coeff_pmm = coeff_data.sqrt_n(n + m + 1)*coeff_data.sqrt_n(n + m + 2);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
            const std::array<double, 2> in_np1_nm1_mm1 = in_np1_nm1[m - 1];
            const std::array<double, 2> in_np1_np1_mm1 = in_np1_np1[m - 1];
            const std::array<double, 2> in_np1_np1_mp1 = in_np1_np1[m + 1];

            /* TODO */
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n_n_m[0]
                    = -nl_coeff_mm*lm_coeff_mmp*in_nm1_nm1_mm1[0]
                    - nl_coeff_pm*lm_coeff_mmp*in_np1_nm1_mm1[0]
                    - nl_coeff_pp*(
                        in_np1_np1_mp1[0] - lm_coeff_pmp*in_np1_np1_mm1[0]);
                
                out_n_n_m[1]
                    = nl_coeff_mm*in_nm1_nm1_mm1[1]
                    + nl_coeff_pm*in_np1_nm1_mm1[1]
                    - nl_coeff_pp*(
                        in_np1_np1_mm1[1] - lm_coeff_pmm*in_np1_np1_mp1[1]);
            }
            else
            {
                out_n_n_m[0]
                    = nl_coeff_pp*(
                        in_np1_np1_mp1[0] + lm_coeff_pmp*in_np1_np1_mm1[0])
                    - nl_coeff_mm*lm_coeff_mmp*in_nm1_nm1_mm1[0]
                    - nl_coeff_pm*lm_coeff_mmp*in_np1_nm1_mm1[0];
                
                out_n_n_m[1]
                    = nl_coeff_mm*in_nm1_nm1_mm1[1]
                    + nl_coeff_pm*in_np1_nm1_mm1[1]
                    - nl_coeff_pp*(
                        in_np1_np1_mm1[1] + lm_coeff_pmm*in_np1_np1_mp1[1]);
            }
        }
    }

    const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(nmax);

    auto out_n = out[nmax];
    auto in_nm1 = in[nmax - 1];
    const std::size_t parity = nmax & 1;

    // edge case: n = nmax, l = 0, m = 0
    /* TODO */
    if (!parity)
    {
        out_n(0,0)[0]
            = (1.0/(sqrt2*sqrt3))*n_coeff_m*double(nmax)*(-in_nm1(1,1)[0] + std::numbers::sqrt2*in_nm1(1,1)[1]);
    }
    else
    {
        out_n(1,0)[0] = 
        out_n(1,1)[0] = 
        out_n(1,1)[1] = 
    }

    for (std::size_t l = 2 + parity; l < nmax; l += 2)
    {
        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
        const double nml = double(nmax - l);
        const double npl1 = double(nmax + l + 1);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;

        auto out_n_l = out_n[l];
        auto in_nm1_lm1 = in_nm1[l - 1];
        auto in_nm1_lp1 = in_nm1[l + 1];

        const double l1_coeff_mmp = coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
        const double l1_coeff_pmp = coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
        const double l1_coeff_mmm = coeff_data.sqrt_n(l - 2)*coeff_data.sqrt_n(l - 1);
        const double l1_coeff_pmm = coeff_data.sqrt_n(l + 2)*coeff_data.sqrt_n(l + 3);

        // edge case: n = nmax, l > 0, m = 0
        // edge case: n = nmax, l > 0, m = 1
        /* TODO */
        if constexpr (COORD == PlaneCoord::X)
        {
            out_n_l[0][0] = (1.0/sqrt2)*(
                    nl_coeff_mm*in_nm1_lm1[1][0]
                    - nl_coeff_mp*in_nm1_lp1[1][0]);
            
            out_n_l[1][0]
                = nl_coeff_mm*(
                    in_nm1_lm1[2][0] - l1_coeff_mmp*sqrt2*in_nm1_lm1[0][0])
                - nl_coeff_mp*(
                    in_nm1_lp1[2][0] - l1_coeff_pmp*sqrt2*in_nm1_lp1[0][0]);
            
            out_n_l[1][1]
                = nl_coeff_mm*(
                    sqrt2*in_nm1_lm1[0][1] - l1_coeff_mmm*in_nm1_lm1[2][1])
                - nl_coeff_mp*(
                    sqrt2*in_nm1_lp1[0][1] - l1_coeff_pmm*in_nm1_lp1[2][1]);
        }
        else
        {
            out_n_l[0][0] = (1.0/sqrt2)*(
                    + nl_coeff_mp*in_nm1_lp1[1][0]
                    - nl_coeff_mm*in_nm1_lm1[1][0]);
            
            out_n_l[1][0]
                = nl_coeff_mp*(
                    in_nm1_lp1[2][0] + l1_coeff_pmp*in_nm1_lp1[0][0])
                - nl_coeff_mm*(
                    in_nm1_lm1[2][0] + l1_coeff_mmp*in_nm1_lm1[0][0]);
            
            out_n_l[1][1]
                = nl_coeff_mm*(
                    in_nm1_lm1[0][1] + l1_coeff_mmm*in_nm1_lm1[2][1])
                - nl_coeff_mp*(
                    in_nm1_lp1[0][1] + l1_coeff_pmm*in_nm1_lp1[2][1]);
        }

        for (std::size_t m = 2; m < l - 1; ++m)
        {
            const double lm_coeff_mmp = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
            const double lm_coeff_pmp = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
            const double lm_coeff_mmm = coeff_data.sqrt_n(l - m - 1)*coeff_data.sqrt_n(l - m);
            const double lm_coeff_pmm = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

            std::array<double, 2>& out_n_l_m = out_n_l[m];
            const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
            const std::array<double, 2> in_nm1_lm1_mp1 = in_nm1_lm1[m + 1];
            const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
            const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];

            // edge case: n = nmax, l > 0, m > 1
            /* TODO */
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n_l_m[0]
                    = nl_coeff_mm*(
                        in_nm1_lm1_mp1[0] - lm_coeff_mmp*in_nm1_lm1_mm1[0])
                    - nl_coeff_mp*(
                        in_nm1_lp1_mp1[0] - lm_coeff_pmp*in_nm1_lp1_mm1[0]);
                
                out_n_l_m[1]
                    = nl_coeff_mm*(
                        in_nm1_lm1_mm1[1] - lm_coeff_mmm*in_nm1_lm1_mp1[1])
                    - nl_coeff_mp*(
                        in_nm1_lp1_mm1[1] - lm_coeff_pmm*in_nm1_lp1_mp1[1]);
            }
            else
            {
                out_n_l_m[0]
                    = nl_coeff_mp*(
                        in_nm1_lp1_mp1[0] + lm_coeff_pmp*in_nm1_lp1_mm1[0])
                    - nl_coeff_mm*(
                        in_nm1_lm1_mp1[0] + lm_coeff_mmp*in_nm1_lm1_mm1[0]);
                
                out_n_l_m[1]
                    = nl_coeff_mm*(
                        in_nm1_lm1_mm1[1] + lm_coeff_mmm*in_nm1_lm1_mp1[1])
                    - nl_coeff_mp*(
                        in_nm1_lp1_mm1[1] + lm_coeff_pmm*in_nm1_lp1_mp1[1]);
            }
        }

        // edge case: n = nmax, l > 0, m = l - 1
        // edge case: n = nmax, l > 0, m = l
        for (std::size_t m = std::max(l - 1, 2UL); m <= l; ++m)
        {
            const double lm_coeff_mmp = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
            const double lm_coeff_pmp = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
            const double lm_coeff_pmm = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

            std::array<double, 2>& out_n_l_m = out_n_l[m];
            const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
            const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
            const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];

            /* TODO */
            if constexpr (COORD == PlaneCoord::X)
            {
                out_n_l_m[0]
                    = -nl_coeff_mm*lm_coeff_mmp*in_nm1_lm1_mm1[0]
                    - nl_coeff_mp*(
                        in_nm1_lp1_mp1[0] - lm_coeff_pmp*in_nm1_lp1_mm1[0]);
                
                out_n_l_m[1]
                    = nl_coeff_mm*in_nm1_lm1_mm1[1]
                    - nl_coeff_mp*(
                        in_nm1_lp1_mm1[1] - lm_coeff_pmm*in_nm1_lp1_mp1[1]);
            }
            else
            {
                out_n_l_m[0]
                    = nl_coeff_mp*(
                        in_nm1_lp1_mp1[0] + lm_coeff_pmp*in_nm1_lp1_mm1[0])
                    - nl_coeff_mm*lm_coeff_mmp*in_nm1_lm1_mm1[0];
                
                out_n_l_m[1]
                    = nl_coeff_mm*in_nm1_lm1_mm1[1]
                    - nl_coeff_mp*(
                        in_nm1_lp1_mm1[1] + lm_coeff_pmm*in_nm1_lp1_mp1[1]);
            }
        }
    }

    const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(nmax);
    const double npl1 = double(2*nmax + 1);
    const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;

    auto out_n_n = out_n[nmax];
    auto in_nm1_nm1 = in_nm1[nmax - 1];

    const double n1_coeff_mmp = coeff_data.sqrt_n(nmax)*coeff_data.sqrt_n(nmax + 1);
    const double n1_coeff_mmm = coeff_data.sqrt_n(nmax - 2)*coeff_data.sqrt_n(nmax - 1);

    // edge case: n = nmax, l = n, m = 0
    // edge case: n = nmax, l = n, m = 1
    /* TODO */
    if constexpr (COORD == PlaneCoord::X)
    {
        out_n_n[0][0] = (1.0/sqrt2)*nl_coeff_mm*in_nm1_nm1[1][0];
        
        out_n_n[1][0]
            = nl_coeff_mm*(
                in_nm1_nm1[2][0] - n1_coeff_mmp*sqrt2*in_nm1_nm1[0][0]);
        
        out_n_n[1][1]
            = nl_coeff_mm*(
                sqrt2*in_nm1_nm1[0][1] - n1_coeff_mmm*in_nm1_nm1[2][1]);
    }
    else
    {
        out_n_n[0][0] = (-1.0/sqrt2)*nl_coeff_mm*in_nm1_nm1[1][0];
        
        out_n_n[1][0]
            = -nl_coeff_mm*(in_nm1_nm1[2][0] + n1_coeff_mmp*in_nm1_nm1[0][0]);
        
        out_n_n[1][1]
            = nl_coeff_mm*(in_nm1_nm1[0][1] + n1_coeff_mmm*in_nm1_nm1[2][1]);
    }
    for (std::size_t m = 2; m < nmax - 1; ++m)
    {
        const double lm_coeff_mmp = coeff_data.sqrt_n(nmax + m - 1)*coeff_data.sqrt_n(nmax + m);
        const double lm_coeff_mmm = coeff_data.sqrt_n(nmax - m - 1)*coeff_data.sqrt_n(nmax - m);

        std::array<double, 2>& out_n_n_m = out_n_n[m];
        const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
        const std::array<double, 2> in_nm1_nm1_mp1 = in_nm1_nm1[m + 1];

        // edge case: n = nmax, l = n, m > 1
        /* TODO */
        if constexpr (COORD == PlaneCoord::X)
        {
            out_n_n_m[0]
                = nl_coeff_mm*(
                    in_nm1_nm1_mp1[0] - lm_coeff_mmp*in_nm1_nm1_mm1[0]);
            
            out_n_n_m[1]
                = nl_coeff_mm*(
                    in_nm1_nm1_mm1[1] - lm_coeff_mmm*in_nm1_nm1_mp1[1]);
        }
        else
        {
            out_n_n_m[0]
                = -nl_coeff_mm*(
                    in_nm1_nm1_mp1[0] + lm_coeff_mmp*in_nm1_nm1_mm1[0]);
            
            out_n_n_m[1]
                = nl_coeff_mm*(
                    in_nm1_nm1_mm1[1] + lm_coeff_mmm*in_nm1_nm1_mp1[1]);
        }
    }

    // edge case: n = nmax, l = n, m = l - 1
    // edge case: n = nmax, l = n, m = l
    for (std::size_t m = std::max(nmax - 1, 2UL); m <= nmax; ++m)
    {
        const double lm_coeff_mmp = coeff_data.sqrt_n(nmax + m - 1)*coeff_data.sqrt_n(nmax + m);

        std::array<double, 2>& out_n_n_m = out_n_n[m];
        const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];

        /* TODO */
        out_n_n_m[0] = -nl_coeff_mm*lm_coeff_mmp*in_nm1_nm1_mm1[0];
        out_n_n_m[1] = nl_coeff_mm*in_nm1_nm1_mm1[1];
    }
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `x`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff_data.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_x(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    multiply_by_x_y_impl<PlaneCoord::X>(coeff_data, in, out);
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `y`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_y(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    multiply_by_x_y_impl<PlaneCoord::Y>(coeff_data, in, out);
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `z`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_z(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    /*
    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 1, l + 1, m)
    in(n + 1, l - 1, m)
    in(n - 1, l + 1, m)
    in(n - 1, l - 1, m)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */
    assert(in.order() < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order();
    if (nmax == 0) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});

    if (nmax > 1)
        out(0,0,0)[0] = (0.5/std::sqrt(5.0))*in(1,1,0)[0];

    for (std::size_t n = 1; n < nmax; ++n)
    {
        const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_p = 0.5*coeff_data.inv_sqrt_2np3_2np5(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        auto in_np1 = in[n + 1];

        const std::size_t parity = n & 1;
        if (!parity)
        {
            out_n(0,0)[0] = (1.0/std::sqrt(3.0))*(
                    n_coeff_m*double(n)*in_nm1(1,0)[0]
                    + n_coeff_p*double(n + 3)*in_np1(1,0)[0]);
        }
        else
        {
            out_n(1,0)[0] = 
            out_n(1,1)[0] = 
            out_n(1,1)[1] = 
        }

        for (std::size_t l = 2 + parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double nml2 = nml + 2.0;
            const double npl3 = npl + 3.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*nml2;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];
            auto in_np1_lm1 = in_np1[l - 1];
            auto in_np1_lp1 = in_np1[l + 1];
            for (std::size_t m = 0; m < l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_m = in_nm1_lm1[m];
                const std::array<double, 2> in_nm1_lp1_m = in_nm1_lp1[m];
                const std::array<double, 2> in_np1_lm1_m = in_np1_lm1[m];
                const std::array<double, 2> in_np1_lp1_m = in_np1_lp1[m];

                out_n_l_m[0]
                    = nl_coeff_mm*coeff_data.sqrt_n(l + m)*in_nm1_lm1_m[0]
                    + nl_coeff_mp*coeff_data.sqrt_n(l - m + 1)*in_nm1_lp1_m[0]
                    + nl_coeff_pm*coeff_data.sqrt_n(l + m)*in_np1_lm1_m[0]
                    + nl_coeff_pp*coeff_data.sqrt_n(l - m + 1)*in_np1_lp1_m[0];

                out_n_l_m[1]
                    = nl_coeff_mm*coeff_data.sqrt_n(l - m)*in_nm1_lm1_m[1]
                    + nl_coeff_mp*coeff_data.sqrt_n(l + m + 1)*in_nm1_lp1_m[1]
                    + nl_coeff_pm*coeff_data.sqrt_n(l - m)*in_np1_lm1_m[1]
                    + nl_coeff_pp*coeff_data.sqrt_n(l + m + 1)*in_np1_lp1_m[1];
            }

            std::array<double, 2>& out_n_l_l = out_n_l[l];
            const std::array<double, 2> in_nm1_lp1_l = in_nm1_lp1[l];
            const std::array<double, 2> in_np1_lp1_l = in_np1_lp1[l];

            out_n_l_l[0]
                = nl_coeff_mp*in_nm1_lp1_l[0] + nl_coeff_pp*in_np1_lp1_l[0];

            out_n_l_l[1]
                = nl_coeff_mp*coeff_data.sqrt_n(2*l + 1)*in_nm1_lp1_l[1]
                + nl_coeff_pp*coeff_data.sqrt_n(2*l + 1)*in_np1_lp1_l[1];
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(n);
        const double npl1 = double(2*n + 1);
        const double npl3 = double(2*n + 3);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_pm = n_coeff_p*l_coeff_m*2.0;
        const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];
        auto in_np1_nm1 = in_np1[n - 1];
        auto in_np1_np1 = in_np1[n + 1];
        for (std::size_t m = 0; m < n; ++m)
        {
            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_m = in_nm1_nm1[m];
            const std::array<double, 2> in_np1_nm1_m = in_np1_nm1[m];
            const std::array<double, 2> in_np1_np1_m = in_np1_np1[m];
            out_n_n_m[0]
                = nl_coeff_mm*coeff_data.sqrt_n(n + m)*in_nm1_nm1_m[0]
                + nl_coeff_pm*coeff_data.sqrt_n(n + m)*in_np1_nm1_m[0]
                + nl_coeff_pp*coeff_data.sqrt_n(n - m + 1)*in_np1_np1_m[0];
            out_n_n_m[1]
                = nl_coeff_mm*coeff_data.sqrt_n(n - m)*in_nm1_nm1_m[1] 
                + nl_coeff_pm*coeff_data.sqrt_n(n - m)*in_np1_nm1_m[1]
                + nl_coeff_pp*coeff_data.sqrt_n(n + m + 1)*in_np1_np1_m[1];
        }

        std::array<double, 2>& out_n_n_n = out_n_n[n];
        const std::array<double, 2> in_np1_np1_n = in_np1_np1[n];
        out_n_n_n[0] = nl_coeff_pp*in_np1_np1_n[0];
        out_n_n_n[1] = nl_coeff_pp*coeff_data.sqrt_n(2*n + 1)*in_np1_np1_n[1];
    }

    const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(nmax);

    auto out_n = out[nmax];
    auto in_nm1 = in[nmax - 1];
    const std::size_t parity = nmax & 1;
    if (!parity)
    {
        out_n(0,0)[0]
            = (1.0/std::sqrt(3.0))*n_coeff_m*double(nmax)*in_nm1(1,0)[0];
    }
    else
    {
        out_n(1,0)[0] = 
        out_n(1,1)[0] = 
        out_n(1,1)[1] = 
    }

    for (std::size_t l = 2 + parity; l < nmax; l += 2)
    {
        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
        const double nml = double(nmax - l);
        const double npl1 = double(nmax + l + 1);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;

        auto out_n_l = out_n[l];
        auto in_nm1_lm1 = in_nm1[l - 1];
        auto in_nm1_lp1 = in_nm1[l + 1];
        for (std::size_t m = 0; m < l; ++m)
        {
            std::array<double, 2>& out_n_l_m = out_n_l[m];
            const std::array<double, 2> in_nm1_lm1_m = in_nm1_lm1[m];
            const std::array<double, 2> in_nm1_lp1_m = in_nm1_lp1[m];

            out_n_l_m[0]
                = nl_coeff_mm*coeff_data.sqrt_n(l + m)*in_nm1_lm1_m[0]
                + nl_coeff_mp*coeff_data.sqrt_n(l - m + 1)*in_nm1_lp1_m[0];

            out_n_l_m[1]
                = nl_coeff_mm*coeff_data.sqrt_n(l - m)*in_nm1_lm1_m[1]
                + nl_coeff_mp*coeff_data.sqrt_n(l + m + 1)*in_nm1_lp1_m[1];
        }

        std::array<double, 2>& out_n_l_l = out_n_l[l];
        const std::array<double, 2> in_nm1_lp1_l = in_nm1_lp1[l];

        out_n_l_l[0] = nl_coeff_mp*in_nm1_lp1_l[0];
        out_n_l_l[1] = nl_coeff_mp*coeff_data.sqrt_n(2*l + 1)*in_nm1_lp1_l[1];
    }

    const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(nmax);
    const double nl_coeff_mm = n_coeff_m*l_coeff_m*double(2*nmax + 1);

    auto out_n_n = out_n[nmax];
    auto in_nm1_nm1 = in_nm1[nmax - 1];
    for (std::size_t m = 0; m < nmax; ++m)
    {
        std::array<double, 2>& out_n_n_m = out_n_n[m];
        const std::array<double, 2> in_nm1_nm1_m = in_nm1_nm1[m];

        out_n_n_m[0] = nl_coeff_mm*coeff_data.sqrt_n(nmax + m)*in_nm1_nm1_m[0];
        out_n_n_m[1] = nl_coeff_mm*coeff_data.sqrt_n(nmax - m)*in_nm1_nm1_m[1];
    }
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `r*r`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in, zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out)
{
    /*
    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 2, l, m)
    in(n, l, m)
    in(n - 2, l, m)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */
    constexpr double sqrt7 = 2.6457513110645905905016158;

    assert(in.order() + 1 < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order() + 1;
    if (nmax == 1) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});

    out(0,0,0)[0] = -(2.0/5.0)*in(0,0,0)[0];
    if (nmax > 3)
    {
        constexpr double coeff = (2.0*std::numbers::sqrt3)/(5.0*sqrt7);
        out(0,0,0)[0] += coeff*in(2,0,0)[0];
    }
    
    if (nmax > 2)
    {
        constexpr double coeff = -(10.0/21.0);
        out(1,1,0)[0] = coeff*in(1,1,0)[0];
        out(1,1,1)[0] = coeff*in(1,1,1)[0];
        out(1,1,1)[1] = coeff*in(1,1,1)[1];
    }

    if (nmax > 4)
    {
        constexpr double coeff = 2.0/(7.0*std::numbers::sqrt3);
        out(1,1,0)[0] += coeff*in(3,1,0)[0];
        out(1,1,1)[0] += coeff*in(3,1,1)[0];
        out(1,1,1)[1] += coeff*in(3,1,1)[1];
    }

    for (std::size_t n = 2; n < nmax - 1; ++n)
    {
        const double _2np1_2np5 = double(2*n + 1)*double(2*n + 5);

        const double n_coeff_m
            = coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_mid = 0.5/_2np1_2np5;
        const double n_coeff_p
            = coeff_data.inv_sqrt_2np3_2np5(n)*coeff_data.inv_sqrt_2np5_2np7(n);
        
        std::size_t parity = n & 1;

        auto out_n = out[n];
        auto in_nm2 = in[n - 2];
        auto in_n = in[n];
        auto in_np2 = in[n + 2];
        for (std::size_t l = parity; l <= n - 2; l += 2)
        {
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double nml2 = nml + 2.0;
            const double npl1 = npl + 1.0;
            const double npl3 = npl + 3.0;
            const double _2lm1 = double(2*l) - 1.0;

            const double nl_coeff_m = nml*npl1*n_coeff_m;
            const double nl_coeff_mid = (_2lm1*_2lm1 - _2np1_2np5)*n_coeff_mid;
            const double nl_coeff_p = nml2*npl3*n_coeff_p;

            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            auto in_n_l = in_n[l];
            auto in_np2_l = in_np2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm2_l_m = in_nm2_l[m];
                const std::array<double, 2> in_n_l_m = in_n_l[m];
                const std::array<double, 2> in_np2_l_m = in_np2_l[m];

                out_n_l_m[0]
                    = nl_coeff_m*in_nm2_l_m[0] + nl_coeff_mid*in_n_l_m[0]
                    + nl_coeff_p*in_np2_l_m[0];
                out_n_l_m[1]
                    = nl_coeff_m*in_nm2_l_m[1] + nl_coeff_mid*in_n_l_m[1]
                    + nl_coeff_p*in_np2_l_m[1];
            }
        }

        const double _2n = double(2*n);
        const double _2nm1 = _2n - 1.0;

        const double nn_coeff_mid = (_2nm1*_2nm1 - _2np1_2np5)*n_coeff_mid;
        const double nn_coeff_p = 2.0*(_2n + 3.0)*n_coeff_p;

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        auto in_np2_n = in_np2[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            std::array<double, 2>& out_n_l_m = out_n_n[m];
            const std::array<double, 2> in_n_l_m = in_n_n[m];
            const std::array<double, 2> in_np2_l_m = in_np2_n[m];

            out_n_l_m[0] = nn_coeff_mid*in_n_l_m[0] + nn_coeff_p*in_np2_l_m[0];
            out_n_l_m[1] = nn_coeff_mid*in_n_l_m[1] + nn_coeff_p*in_np2_l_m[1];
        }
    }

    for (std::size_t n = nmax - ((nmax == 2) ? 0 : 1); n <= nmax; ++n)
    {
        const double _2np1_2np5 = double(2*n + 1)*double(2*n + 5);

        const double n_coeff_m
            = coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_mid = 0.5/_2np1_2np5;

        std::size_t parity = n & 1;

        auto out_n = out[n];
        auto in_nm2 = in[n - 2];
        auto in_n = in[n];
        for (std::size_t l = parity; l <= n - 2; l += 2)
        {
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double _2lm1 = double(2*l) - 1.0;

            const double nl_coeff_m = nml*npl1*n_coeff_m;
            const double nl_coeff_mid = (_2lm1*_2lm1 - _2np1_2np5)*n_coeff_mid;

            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            auto in_n_l = in_n[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm2_l_m = in_nm2_l[m];
                const std::array<double, 2> in_n_l_m = in_n_l[m];

                out_n_l_m[0]
                    = nl_coeff_m*in_nm2_l_m[0] + nl_coeff_mid*in_n_l_m[0];
                out_n_l_m[1]
                    = nl_coeff_m*in_nm2_l_m[1] + nl_coeff_mid*in_n_l_m[1];
            }
        }

        const double _2n = double(2*n);
        const double _2nm1 = _2n - 1.0;

        const double nn_coeff_mid = (_2nm1*_2nm1 - _2np1_2np5)*n_coeff_mid;

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            std::array<double, 2>& out_n_l_m = out_n_n[m];
            const std::array<double, 2> in_n_l_m = in_n_n[m];

            out_n_l_m[0] = nn_coeff_mid*in_n_l_m[0];
            out_n_l_m[1] = nn_coeff_mid*in_n_l_m[1];
        }
    }
}

} // detail
} // zebra
