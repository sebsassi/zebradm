#include <random>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"
#include "nanobench.h"
#include "distributions.hpp"

void benchmark_distribution_zernike_grid_construction(
    ankerl::nanobench::Bench& bench, const char* name,
    DistributionCartesian dist, std::size_t order)
{
    zest::zt::BallGLQGridPoints points(order);
    zest::zt::BallGLQGrid<double> grid(order);
    bench.run(name, [&](){
        points.generate_values(grid, dist);
    });
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, std::span<const std::size_t> orders)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));
    bench.maxEpochTime(std::chrono::nanoseconds(100000000000));

    bench.title("zest::zt::BallGLQGridPoints::generate_values");
    for (std::size_t order : orders)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", order);
        benchmark_distribution_zernike_grid_construction(
                bench, name, dist, order);
    }

    char fname[512] = {};
    std::sprintf(fname, "distribution_grid_construction_bench_%s_%.2f_%lu_%lu.json", dist_name);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}

int main([[maybe_unused]] int argc, char** argv)
{
    const std::size_t bench_ind = atoi(argv[1]);
    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200};
    switch (bench_ind)
    {
        case 0:
            run_benchmarks(aniso_gaussian, "aniso_gaussian", orders);
            break;
        case 1:
            run_benchmarks(four_gaussians, "four_gaussians", orders);
            break;
        case 2:
            run_benchmarks(shm_plus_stream, "shm_plus_stream", orders);
            break;
        case 3:
            run_benchmarks(shmpp_aniso, "shmpp_aniso", orders);
            break;
        case 4:
            run_benchmarks(shmpp, "shmpp", orders);
            break;
    }
}