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
#include "responses.hpp"

constexpr double relative_error(double test, double ref)
{
    return std::fabs(1.0 - test/ref);
}

void benchmark_radon_angle_integrator_anisotropic(
    ankerl::nanobench::Bench& bench, const char* name, DistributionCartesian dist, Response resp, const std::span<const std::array<double, 3>> boosts, std::span<const double> eras, std::span<const double> min_speeds, double relerr, std::size_t max_subdiv)
{
    std::vector<double> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    bench.run(name, [&](){
        integrator.integrate(
                dist, resp, boosts, eras, min_speeds, 0.0, relerr, out, max_subdiv);
    });
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, Response resp, const char* resp_name, std::span<const double> relerrs, std::span<const std::size_t> max_subdiv, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds, double time_limit_lo_s, double time_limit_hi_s)
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

    std::vector<double> eras(num_boosts);
    for (auto& element : eras)
        element = 2.0*std::numbers::pi*rng_dist(gen);

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*(boost_len + 1.0)/double(num_min_speeds - 1);

    std::printf("Begin benchmarks\n");
    bench.title("integrate::RadonAngleIntegrator::integrate");
    bool soft_break = false;
    for (std::size_t i = 0; i < relerrs.size(); ++i)
    {
        char name[32] = {};
        std::sprintf(name, "%.2e", relerrs[i]);
        benchmark_radon_angle_integrator_anisotropic(
                bench, name, dist, resp, boosts, eras, min_speeds, relerrs[i], max_subdiv[i]);
        const double elapsed_s = bench.results()[bench.results().size() - 1]
            .sumProduct(
                ankerl::nanobench::Result::Measure::iterations, ankerl::nanobench::Result::Measure::elapsed);
        if (elapsed_s >= time_limit_hi_s || soft_break) break;
        if (elapsed_s >= time_limit_lo_s) soft_break = true;
    }

    char fname[512] = {};
    std::sprintf(fname, "radon_anisotropic_angle_integrator_bench_%s_%s_%.2f_%lu_%lu.json", dist_name, resp_name, boost_len, num_boosts, num_min_speeds);

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

    constexpr std::array<Labeled<Response>, 2> responses = {
        Labeled<Response>{smooth_exponential, "smooth_exponential"},
        Labeled<Response>{smooth_dots, "smooth_dots"}
    };

    if (argc < 8)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   dist_ind:           index of distribution {0,1,2,3,4}\n"
            "   resp_ind:           index of response {0,1}\n"
            "   boost_len:          length of boost vector (float)\n"
            "   num_boosts:         number of boost vectors (positive integer)\n"
            "   num_min_speeds:     number of min_speed values (positive integer)\n"
            "   time_limit_lo_s:    shoft time cutoff in seconds (positive integer)\n"
            "   time_limit_hi_s:    hard time cutoff in seconds (positive integer)");

    const std::size_t dist_ind = atoi(argv[1]);
    const std::size_t resp_ind = atoi(argv[2]);
    const double boost_len = atof(argv[3]);
    const std::size_t num_boosts = atoi(argv[4]);
    const std::size_t num_min_speeds = atoi(argv[5]);
    const double time_limit_lo_s = double(atoi(argv[6]));
    const double time_limit_hi_s = double(atoi(argv[7]));
    std::printf("dist_ind: %lu\nresp_ind: %lu\nboost_len: %.2f\nnum_boosts: %lu\nnum_speeds: %lu\ntime_limit_lo_s: %.0f\ntime_limit_hi_s: %.0f\n", dist_ind, resp_ind, boost_len, num_boosts, num_min_speeds, time_limit_lo_s, time_limit_hi_s);

    const std::vector<double> relerrs = {
        /*1.0e+2, 1.0e+1, 1.0e+0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,*/ 1.0e-7/*, 1.0e-8, 1.0e-9, 1.0e-10*/
    };

    const std::vector<std::size_t> max_subdiv = {
        1000000/*, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000*/
    };

    const Labeled<DistributionCartesian> dist = distributions[dist_ind];
    const Labeled<Response> resp = responses[resp_ind];

    run_benchmarks(
            dist.object, dist.label, resp.object, resp.label, relerrs, max_subdiv, boost_len, num_boosts, num_min_speeds, time_limit_lo_s, time_limit_hi_s);
}