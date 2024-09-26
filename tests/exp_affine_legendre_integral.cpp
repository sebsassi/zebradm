#include "affine_legendre_integral.hpp"

#include <fstream>
#include <cmath>

void exp_affine_legendre_integral_recursion_stability(
    double shift, double scale)
{
    constexpr std::size_t order = 100;
    constexpr std::size_t extra_extent = 0;

    zebra::AffineLegendreIntegrals recursion(order, extra_extent);

    std::vector<double> trapezoid_buffer(
        zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    
    zebra::TrapezoidSpan<double> test_trapezoid(
        trapezoid_buffer.data(), order, extra_extent);
    recursion.integrals(test_trapezoid, shift, scale);

    char fname[128] = {};
    std::sprintf(fname, "aff_leg_coeff_%.2f_%.2f.dat", shift, scale);
    std::ofstream output{};
    output.open(fname);
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = 0; l <= n; ++l)
        {
            char line[128] = {};
            std::sprintf(line, "%.16e ", std::fabs(test_trapezoid(n, l)));
            output << line;
        }
        for (std::size_t l = n + 1; l < order; ++l)
        {
            char line[128] = {};
            std::sprintf(line, "%.16e ", 0.0);
            output << line;
        }
        output << '\n';
    }
    output.close();
}

int main()
{
    std::vector<double> shifts = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5};
    std::vector<double> scales = {0.25, 0.5, 0.75, 1.0};

    for (auto shift : shifts)
    {
        for (auto scale : scales)
        {
            exp_affine_legendre_integral_recursion_stability(shift, scale);
        }
    }
}