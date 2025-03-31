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

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"
#include "radon_integrator.hpp"
#include "nanobench.h"
#include "distributions.hpp"

void benchmark_radon_angle_integrator_isotropic(
    ankerl::nanobench::Bench& bench, const char* name, DistributionCartesian dist, const char* dist_name, std::span<const std::array<double, 3>> boosts, std::span<const double > min_speeds, double relerr, zest::MDSpan<const double, 2> reference)
{
    std::vector<double> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    std::size_t max_subdiv = 200000000;
    zdm::integrate::RadonAngleIntegrator integrator{};
    bench.run(name, [&](){
        integrator.integrate(
                dist, boosts, min_speeds, 0.0, relerr, out, max_subdiv);
    });

    char fname[512] = {};
    std::sprintf(fname, "radon_isotropic_angle_integrator_error_relative_%s_%.2e.dat", dist_name, relerr);
    std::ofstream output{};
    output.open(fname);
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            double error = 0.0;
            if (reference(i, j) != 0.0)
            {
                error = std::fabs(1.0 - out(i, j)/reference(i, j));
            }
            output << error << ' ';
        }
        output << '\n';
    }
    output.close();
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, std::span<const double> relerrs, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds, double time_limit_s)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));
    bench.maxEpochTime(std::chrono::nanoseconds(100000000000));

    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> boosts(num_boosts);
    for (auto& element : boosts)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        element = {
            boost_len*st*std::cos(az), boost_len*st*std::sin(az), ct
        };
    }

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*(boost_len + 1.0)/double(num_min_speeds - 1);

    constexpr std::size_t reference_order = 200;
    zest::zt::RealZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_order).transform(
            dist, 1.0, reference_order);
    
    std::vector<double> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zdm::zebra::IsotropicAngleIntegrator integrator(reference_order);
    integrator.integrate(reference_distribution, boosts, min_speeds, reference);

    bench.title("integrate::RadonAngleIntegrator::integrate");
    for (double relerr : relerrs)
    {
        char name[32] = {};
        std::sprintf(name, "%.16e", relerr);
        benchmark_radon_angle_integrator_isotropic(
                bench, name, dist, dist_name, boosts, min_speeds, relerr, reference);
        const double elapsed_s = bench.results()[bench.results().size() - 1]
            .sumProduct(
                ankerl::nanobench::Result::Measure::iterations, ankerl::nanobench::Result::Measure::elapsed);
        if (elapsed_s >= time_limit_s) break;
    }

    char fname[512] = {};
    std::sprintf(fname, "radon_angle_integrator_isotropic_bench_%s_%.2f_%lu_%lu.json", dist_name, boost_len, num_boosts, num_min_speeds);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr std::array<Labeled<DistributionCartesian>, 5> distributions = {
        Labeled<DistributionCartesian>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionCartesian>{four_gaussians, "four_gaussians"},
        Labeled<DistributionCartesian>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionCartesian>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionCartesian>{shmpp, "shmpp"}
    };

    if (argc < 6)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   dist_ind:       index of distribution {0,1,2,3,4}\n"
            "   boost_len:      length of boost vector (float)\n"
            "   num_boosts:     number of boost vectors (positive integer)\n"
            "   num_min_speeds: number of min_speed values (positive integer)\n"
            "   time_limit_s:   hard time cutoff in seconds (positive integer)");

    const std::size_t dist_ind = atoi(argv[1]);
    const double boost_len = atof(argv[2]);
    const std::size_t num_boosts = atoi(argv[3]);
    const std::size_t num_min_speeds = atoi(argv[4]);
    const double time_limit_s = double(atoi(argv[5]));

    const std::vector<double> relerrs = {
        1.0e+2, 1.0e+1, 1.0e+0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10
    };

    const Labeled<DistributionCartesian> dist = distributions[dist_ind];
    
    run_benchmarks(
            dist.object, dist.label, relerrs, boost_len, num_boosts, num_min_speeds, time_limit_s);
}