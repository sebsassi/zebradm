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

#include "zebra_angle_integrator.hpp"
#include "types.hpp"
#include "nanobench.h"

void benchmark_zebra_anisotropic_angle_integrator(
    ankerl::nanobench::Bench& bench, const char* name, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds, std::size_t dist_order, std::size_t resp_order)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::ZernikeExpansionOrthoGeo distribution(dist_order);
    for (auto& element : distribution.flatten())
        element = {dist(gen), dist(gen)};
    
    std::vector<std::array<double, 2>> response_buffer(SHExpansionCollectionSpan<std::array<double, 2>>::size(num_min_speeds, resp_order));
    for (auto& element : response_buffer)
        element = {dist(gen), dist(gen)};
    SHExpansionCollectionSpan<const std::array<double, 2>> response(
            response_buffer, num_min_speeds, resp_order);

    std::vector<std::array<double, 3>> boosts(num_boosts);
    for (auto& element : boosts)
    {
        const double ct = 2.0*dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*dist(gen);
        element = {
            boost_len*st*std::cos(az), boost_len*st*std::sin(az), ct
        };
    }

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*(boost_len + 1.0)/double(num_min_speeds - 1);

    std::vector<double> eras(num_boosts);
    for (auto& element : eras)
        element = 2.0*std::numbers::pi*dist(gen);

    std::vector<double> out_buffer(num_boosts*num_min_speeds);
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);
    bench.run(name, [&](){
        integrator.integrate(
                distribution, response, boosts, eras, min_speeds, out);
    });
}

int main([[maybe_unused]] int argc, char** argv)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    const double boost_len = atof(argv[1]);
    const std::size_t num_boosts = atoi(argv[2]);
    const std::size_t num_min_speeds = atoi(argv[3]);
    const std::size_t resp_order = atoi(argv[4]);

    std::vector<std::size_t> dist_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200};

    bench.title("zebra::AnisotropicAngleIntegrator::integrate");
    for (const auto& dist_order : dist_orders)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", dist_order);
        benchmark_zebra_anisotropic_angle_integrator(
                bench, name, boost_len, num_boosts, num_min_speeds, dist_order, resp_order);
    }

    char fname[128] = {};
    std::sprintf(fname, "zebra_anisotropic_angle_integrator_bench_%.2f_%lu_%lu_%lu.json", boost_len, num_boosts, num_min_speeds, resp_order);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}