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

#include <numbers>
#include <print>
#include <random>
#include <vector>

#include "coordinate_transforms.hpp"
#include "transform_utilities.hpp"
#include "types.hpp"

namespace
{

bool is_close(double a, double b, double error)
{
    return std::abs(a - b) < error;
}

bool test_from_points_is_correct_for_Z000(double radius)
{
    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat, [[maybe_unused]] double r)
    {
        return std::numbers::sqrt3;
    };

    const std::size_t count = 2000000;
    std::vector<zdm::la::Vector<double, 3>> points(count);
    std::vector<double> values(count);

    std::mt19937 rng{1337};
    std::uniform_real_distribution<double> dist{};

    for (std::size_t i = 0; i < count; ++i)
    {
        const double lon = 2.0*std::numbers::pi*dist(rng);
        const double colat = std::acos(2.0*dist(rng) - 1.0);
        const double r = std::cbrt(radius*dist(rng));

        points[i] = zdm::coordinates::spherical_to_cartesian_phys(lon, colat, r);
        values[i] = function(lon, colat, r);
    }

    zdm::ZernikeExpansion expansion = zdm::from_points(points, values, 10);

    constexpr double tol = 2.0e-3;
    constexpr double reference_coeff = 1.0;

    bool success = is_close(expansion[0, 0, 0, 0], reference_coeff, tol);
    for (auto n : expansion.indices(1))
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                success = success && is_close(expansion_nl[m, 0], 0.0, tol);
                success = success && is_close(expansion_nl[m, 1], 0.0, tol);
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                    std::println("{}, {}, {}: {:.16e} {:.16e}", n, l, m, expansion_nl[m, 0], expansion_nl[m, 1]);
            }
        }
    }

    return success;
}

} // namespace

int main()
{
    assert(test_from_points_is_correct_for_Z000(1.0));
    assert(test_from_points_is_correct_for_Z000(2.0));
}
