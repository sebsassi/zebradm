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

#include <fstream>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"

#include "distributions.hpp"

namespace
{

void isotropic_convergence(
    const std::array<double, 3>& offset, std::span<const double> shells,
    std::span<double> reference, DistributionSpherical dist, const char* name,
    std::size_t order)
{
    zdm::ZernikeExpansion distribution
        = zest::zt::ZernikeTransformerNormalGeo(order).forward_transform(
            dist, 1.0, order);

    std::vector<double> out(shells.size());

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso> integrator(order);
    integrator.integrate(distribution, offset, shells, out);

    char fname[512] = {};
    std::sprintf(fname, "zebra_convergence_demo_%s_order_%lu.dat", name, order);
    std::ofstream output{};
    output.open(fname);
    for (std::size_t i = 0; i < shells.size(); ++i)
    {
        const double absolute_error = std::fabs(reference[i] - out[i]);
        output << shells[i] << ' ' << out[i] << ' ' << absolute_error  << '\n';
    }
    output.close();
}

void convergence_demo(
    DistributionSpherical dist, const char* name)
{
    constexpr std::size_t reference_order = 200;

    // constexpr std::size_t num_offsets = 100;
    constexpr std::size_t num_shells = 100;

    constexpr double max_shell = 1.5;

    constexpr std::array<double, 3> offset = {0.5, 0.0, 0.0};

    std::vector<double> shells(num_shells);
    for (std::size_t i = 0; i < num_shells; ++i)
        shells[i] = double(i)*max_shell/double(num_shells - 1);

    zdm::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerNormalGeo(reference_order).forward_transform(
            dist, 1.0, reference_order);

    std::vector<double> reference(shells.size());

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso>
    integrator(reference_order);

    integrator.integrate(reference_distribution, offset, shells, reference);

    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};

    for (std::size_t order : orders)
        isotropic_convergence(offset, shells, reference, dist, name, order);
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

} // namespace

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr std::array<Labeled<DistributionSpherical>, 5> distributions = {
        Labeled<DistributionSpherical>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionSpherical>{four_gaussians, "four_gaussians"},
        Labeled<DistributionSpherical>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionSpherical>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionSpherical>{shmpp, "shmpp"}
    };

    const std::size_t dist_ind = std::size_t(atoi(argv[1]));
    const Labeled<DistributionSpherical> dist = distributions[dist_ind];

    convergence_demo(dist.object, dist.label);
}
