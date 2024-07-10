#include "radon_util.hpp"

#include <cassert>
#include <cstdio>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::max(std::fabs(a[0] - b[0]), std::fabs(a[1] - b[1])) < tol;
}

bool test_gegenbauer_recursion_accepts_order(std::size_t order)
{
    zest::zt::ZernikeExpansionOrthoGeo in(order);
    zest::zt::ZernikeExpansionOrthoGeo out(order + 2);
    detail::apply_gegenbauer_recursion(in, out);
    return true;
}

bool test_gegenbauer_recursion_is_correct_to_order_5()
{
    static constexpr zest::zt::ZernikeNorm NORM = zest::zt::ZernikeNorm::NORMED;
    constexpr std::size_t order = 5;
    zest::zt::ZernikeExpansionOrthoGeo in(order);
    in(0,0,0) = {1.0, -1.0};
    in(1,1,0) = {2.0, -2.0};
    in(1,1,1) = {3.0, -3.0};
    in(2,0,0) = {4.0, -4.0};
    in(2,2,0) = {5.0, -5.0};
    in(2,2,1) = {6.0, -6.0};
    in(2,2,2) = {7.0, -7.0};
    in(3,1,0) = {8.0, -8.0};
    in(3,1,1) = {9.0, -9.0};
    in(3,3,0) = {10.0, -10.0};
    in(3,3,1) = {11.0, -11.0};
    in(3,3,2) = {12.0, -12.0};
    in(3,3,3) = {13.0, -13.0};
    in(4,0,0) = {14.0, -14.0};
    in(4,2,0) = {15.0, -15.0};
    in(4,2,1) = {16.0, -16.0};
    in(4,2,2) = {17.0, -17.0};
    in(4,4,0) = {18.0, -18.0};
    in(4,4,1) = {19.0, -19.0};
    in(4,4,2) = {20.0, -20.0};
    in(4,4,3) = {21.0, -21.0};
    in(4,4,4) = {22.0, -22.0};

    zest::zt::ZernikeExpansionOrthoGeo out_ref(order + 2);
    out_ref(0,0,0) = detail::geg_rec_coeff<NORM>(0)*in(0,0,0);
    out_ref(1,1,0) = detail::geg_rec_coeff<NORM>(1)*in(1,1,0);
    out_ref(1,1,1) = detail::geg_rec_coeff<NORM>(1)*in(1,1,1);
    out_ref(2,0,0) = detail::geg_rec_coeff<NORM>(2)*in(2,0,0)
                        - detail::geg_rec_coeff<NORM>(0)*in(0,0,0);
    out_ref(2,2,0) = detail::geg_rec_coeff<NORM>(2)*in(2,2,0);
    out_ref(2,2,1) = detail::geg_rec_coeff<NORM>(2)*in(2,2,1);
    out_ref(2,2,2) = detail::geg_rec_coeff<NORM>(2)*in(2,2,2);
    out_ref(3,1,0) = detail::geg_rec_coeff<NORM>(3)*in(3,1,0)
                        - detail::geg_rec_coeff<NORM>(1)*in(1,1,0);
    out_ref(3,1,1) = detail::geg_rec_coeff<NORM>(3)*in(3,1,1)
                        - detail::geg_rec_coeff<NORM>(1)*in(1,1,1);
    out_ref(3,3,0) = detail::geg_rec_coeff<NORM>(3)*in(3,3,0);
    out_ref(3,3,1) = detail::geg_rec_coeff<NORM>(3)*in(3,3,1);
    out_ref(3,3,2) = detail::geg_rec_coeff<NORM>(3)*in(3,3,2);
    out_ref(3,3,3) = detail::geg_rec_coeff<NORM>(3)*in(3,3,3);
    out_ref(4,0,0) = detail::geg_rec_coeff<NORM>(4)*in(4,0,0)
                        - detail::geg_rec_coeff<NORM>(2)*in(2,0,0);
    out_ref(4,2,0) = detail::geg_rec_coeff<NORM>(4)*in(4,2,0)
                        - detail::geg_rec_coeff<NORM>(2)*in(2,2,0);
    out_ref(4,2,1) = detail::geg_rec_coeff<NORM>(4)*in(4,2,1)
                        - detail::geg_rec_coeff<NORM>(2)*in(2,2,1);
    out_ref(4,2,2) = detail::geg_rec_coeff<NORM>(4)*in(4,2,2)
                        - detail::geg_rec_coeff<NORM>(2)*in(2,2,2);
    out_ref(4,4,0) = detail::geg_rec_coeff<NORM>(4)*in(4,4,0);
    out_ref(4,4,1) = detail::geg_rec_coeff<NORM>(4)*in(4,4,1);
    out_ref(4,4,2) = detail::geg_rec_coeff<NORM>(4)*in(4,4,2);
    out_ref(4,4,3) = detail::geg_rec_coeff<NORM>(4)*in(4,4,3);
    out_ref(4,4,4) = detail::geg_rec_coeff<NORM>(4)*in(4,4,4);
    out_ref(5,1,0) = -detail::geg_rec_coeff<NORM>(3)*in(3,1,0);
    out_ref(5,1,1) = -detail::geg_rec_coeff<NORM>(3)*in(3,1,1);
    out_ref(5,3,0) = -detail::geg_rec_coeff<NORM>(3)*in(3,3,0);
    out_ref(5,3,1) = -detail::geg_rec_coeff<NORM>(3)*in(3,3,1);
    out_ref(5,3,2) = -detail::geg_rec_coeff<NORM>(3)*in(3,3,2);
    out_ref(5,3,3) = -detail::geg_rec_coeff<NORM>(3)*in(3,3,3);
    out_ref(5,5,0) = {0.0, 0.0};
    out_ref(5,5,1) = {0.0, 0.0};
    out_ref(5,5,2) = {0.0, 0.0};
    out_ref(5,5,3) = {0.0, 0.0};
    out_ref(5,5,4) = {0.0, 0.0};
    out_ref(5,5,5) = {0.0, 0.0};
    out_ref(6,0,0) = -detail::geg_rec_coeff<NORM>(4)*in(4,0,0);
    out_ref(6,2,0) = -detail::geg_rec_coeff<NORM>(4)*in(4,2,0);
    out_ref(6,2,1) = -detail::geg_rec_coeff<NORM>(4)*in(4,2,1);
    out_ref(6,2,2) = -detail::geg_rec_coeff<NORM>(4)*in(4,2,2);
    out_ref(6,4,0) = -detail::geg_rec_coeff<NORM>(4)*in(4,4,0);
    out_ref(6,4,1) = -detail::geg_rec_coeff<NORM>(4)*in(4,4,1);
    out_ref(6,4,2) = -detail::geg_rec_coeff<NORM>(4)*in(4,4,2);
    out_ref(6,4,3) = -detail::geg_rec_coeff<NORM>(4)*in(4,4,3);
    out_ref(6,4,4) = -detail::geg_rec_coeff<NORM>(4)*in(4,4,4);
    out_ref(6,6,0) = {0.0, 0.0};
    out_ref(6,6,1) = {0.0, 0.0};
    out_ref(6,6,2) = {0.0, 0.0};
    out_ref(6,6,3) = {0.0, 0.0};
    out_ref(6,6,4) = {0.0, 0.0};
    out_ref(6,6,5) = {0.0, 0.0};
    out_ref(6,6,6) = {0.0, 0.0};

    zest::zt::ZernikeExpansionOrthoGeo out(order + 2);

    detail::apply_gegenbauer_recursion(in, out);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success && (is_close(out(n,l,m), out_ref(n,l,m), tol));
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf("(%lu,%lu,%lu): {%f, %f} {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1], out_ref(n,l,m)[0], out_ref(n,l,m)[1]);
            }
        }
    }

    return success;
}

int main()
{
    assert(test_gegenbauer_recursion_accepts_order(0));
    assert(test_gegenbauer_recursion_accepts_order(1));
    assert(test_gegenbauer_recursion_accepts_order(2));
    assert(test_gegenbauer_recursion_accepts_order(3));
    assert(test_gegenbauer_recursion_is_correct_to_order_5());
}