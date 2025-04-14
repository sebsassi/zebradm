/*
Copyright (c) 2024 Sebastian Sassi

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
#include <array>
#include <cmath>
#include <numbers>

#include "linalg.hpp"
#include "coordinate_transforms.hpp"

using Response = double(*)(double, double, double);

double smooth_exponential(double shell, double longitude, double colatitude)
{
    static const std::array<double, 3> ref_dir
        = zdm::normalize(std::array<double, 3>{0.5, 0.5, 0.5});
    const std::array<double, 3> dir
        = zdm::coordinates::spherical_to_cartesian_phys(longitude, colatitude);
    constexpr double rate = 2.0;
    const double u2 = shell*shell;
    const double u4 = u2*u2;
    return (u4/(1 + u4))*std::exp(rate*(zdm::dot(dir, ref_dir)));
}

inline double smooth_step(double x, double slope)
{
    return 0.5*(1.0 + std::tanh(slope*x));
}

double smooth_dots(double shell, double longitude, double colatitude)
{
    constexpr double norm = (4000.0/(3.0*33.0*33.0));

    const double t = std::cos(colatitude);
    const double t2 = t*t;
    const double u2 = (1.0 - t)*(1.0 + t);
    const double u4 = u2*u2;
    const double Y64 = norm*u4*(11.0*t2 - 1.0)*std::cos(4.0*longitude);

    constexpr double rate = 2.0;
    const double surface = 1.0 - std::exp(rate*(Y64 - 1.0));

    constexpr double slope = 10.0;
    return smooth_step(shell*(1.0/1.5) - surface, slope);
}

double fcc_dots(double shell, double longitude, double colatitude)
{
    static const std::array<std::array<double, 3>, 14> ref_dirs = {
        std::array<double, 3>{1.0, 0.0, 0.0},
        std::array<double, 3>{-1.0, 0.0, 0.0},
        std::array<double, 3>{0.0, 1.0, 0.0},
        std::array<double, 3>{0.0, -1.0, 0.0},
        std::array<double, 3>{0.0, 0.0, 1.0},
        std::array<double, 3>{0.0, 0.0, -1.0},
        std::array<double, 3>{1.0, 1.0, 1.0},
        std::array<double, 3>{1.0, 1.0, -1.0},
        std::array<double, 3>{1.0, -1.0, 1.0},
        std::array<double, 3>{1.0, -1.0, -1.0},
        std::array<double, 3>{-1.0, 1.0, 1.0},
        std::array<double, 3>{-1.0, 1.0, -1.0},
        std::array<double, 3>{-1.0, -1.0, 1.0},
        std::array<double, 3>{-1.0, -1.0, -1.0}
    };
    
    const std::array<double, 3> dir
        = zdm::coordinates::spherical_to_cartesian_phys(longitude, colatitude);
    
    const double radius = std::acos((1.0/1.5)*shell);
    const double c_rad = std::cos(radius);

    double res = 0.0;
    for (const auto& ref_dir : ref_dirs)
        res += double(zdm::dot(dir, ref_dir) < c_rad);
    
    return res;
}
