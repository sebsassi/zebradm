#include <random>
#include <fstream>

#include "zebra_angle_integrator.hpp"
#include "nanobench.h"

void benchmark_zebra_isotropic_angle_integrator_transverse(
    ankerl::nanobench::Bench& bench, const char* name, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds, std::size_t order)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::ZernikeExpansionOrthoGeo distribution(order);
    for (auto& element : distribution.flatten())
        element = {dist(gen), dist(gen)};

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

    std::vector<std::array<double, 2>> out_buffer(num_boosts*num_min_speeds);
    zest::MDSpan<std::array<double, 2>, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zebra::IsotropicTransverseAngleIntegrator integrator(order);
    bench.run(name, [&](){
        integrator.integrate(
                distribution, boosts, min_speeds, out);
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

    std::vector<std::size_t> order_vec = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200};

    bench.title("zebra::IsotropicTransverseAngleIntegrator::integrate");
    for (const auto& order : order_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", order);
        benchmark_zebra_isotropic_angle_integrator_transverse(
                bench, name, boost_len, num_boosts, num_min_speeds, order);
    }

    char fname[128] = {};
    std::sprintf(fname, "zebra_isotropic_transverse_angle_integrator_bench_%.2f_%lu_%lu.json", boost_len, num_boosts, num_min_speeds);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}