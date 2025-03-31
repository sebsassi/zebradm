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
#include <random>
#include <fstream>
#include <iomanip>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"

#include "distributions.hpp"

void angle_integrator_error(
    std::span<const std::array<double, 3>> boosts, std::span<const double > min_speeds, zest::MDSpan<double, 2> reference, DistributionSpherical dist, const char* name, std::size_t order, bool relative_error)
{
    zest::zt::RealZernikeExpansion distribution
        = zest::zt::ZernikeTransformerOrthoGeo(order).transform(
            dist, 1.0, order);

    std::vector<double> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zdm::zebra::IsotropicAngleIntegrator integrator(order);
    integrator.integrate(distribution, boosts, min_speeds, out);

    char fname[512] = {};
    if (relative_error)
        std::sprintf(fname, "zebra_angle_integrator_error_relative_%s_order_%lu.dat", name, order);
    else
        std::sprintf(fname, "zebra_angle_integrator_error_absolute_%s_order_%lu.dat", name, order);
    std::ofstream output{};
    output.open(fname);
    output << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            double error = 0.0;
            if (reference(i, j) != 0.0)
            {
                error = (relative_error) ?
                    std::fabs(1.0 - out(i, j)/reference(i, j))
                    : std::fabs(reference(i, j) - out(i, j));
            }
            output << error << ' ';
        }
        output << '\n';
    }
    output.close();
}

void angle_integrator_errors(
    DistributionSpherical dist, const char* name, bool relative_error,
    double boost_len, std::size_t num_boosts, std::size_t num_min_speeds)
{
    constexpr std::size_t reference_order = 200;

    const double max_min_speed = 1.0 + boost_len;

    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> boosts(num_boosts);
    for (std::size_t i = 0; i < num_boosts; ++i)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        boosts[i] = {
            boost_len*st*std::cos(az), boost_len*st*std::sin(az), ct
        };
    }

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*max_min_speed/double(num_min_speeds - 1);

    zest::zt::RealZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_order).transform(
            dist, 1.0, reference_order);
    
    std::vector<double> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zdm::zebra::IsotropicAngleIntegrator integrator(reference_order);
    integrator.integrate(reference_distribution, boosts, min_speeds, reference);

    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};

    for (std::size_t order : orders)
        angle_integrator_error(
                boosts, min_speeds, reference, dist, name, order, 
                relative_error);
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    const bool relative_error = true;
    constexpr std::array<Labeled<DistributionSpherical>, 5> distributions = {
        Labeled<DistributionSpherical>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionSpherical>{four_gaussians, "four_gaussians"},
        Labeled<DistributionSpherical>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionSpherical>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionSpherical>{shmpp, "shmpp"}
    };

    if (argc < 5)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   dist_ind:       index of distribution {0,1,2,3,4}\n"
            "   boost_len:      length of boost vector (float)\n"
            "   num_boosts:     number of boost vectors (positive integer)\n"
            "   num_min_speeds: number of min_speed values (positive integer)");

    const std::size_t dist_ind = atoi(argv[1]);
    const double boost_len = atof(argv[2]);
    const std::size_t num_boosts = atoi(argv[3]);
    const std::size_t num_min_speeds = atoi(argv[4]);

    const Labeled<DistributionSpherical> dist = distributions[dist_ind];
    angle_integrator_errors(
            dist.object, dist.label, relative_error, boost_len, num_boosts, num_min_speeds);
}