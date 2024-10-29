#include <random>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"
#include "radon_integrator.hpp"
#include "nanobench.h"
#include "distributions.hpp"

constexpr std::array<double, 2> relative_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(1.0 - test[0]/ref[0]),
        std::fabs(1.0 - test[1]/ref[1])
    };
}

void benchmark_radon_angle_integrator_isotropic(
    ankerl::nanobench::Bench& bench, const char* name, DistributionCartesian dist, const char* dist_name, std::span<const std::array<double, 3>> boosts, std::span<const double> min_speeds, double relerr, zest::MDSpan<const std::array<double, 2>, 2> reference)
{
    std::vector<std::array<double, 2>> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    std::size_t max_subdiv = 200000000;
    integrate::RadonAngleIntegrator integrator{};
    bench.run(name, [&](){
        integrator.integrate_transverse(
                dist, boosts, min_speeds, 0.0, relerr, out, max_subdiv);
    });

    char fname_nt[512] = {};
    char fname_t[512] = {};
    std::sprintf(fname_nt, "radon_angle_integrator_nontransverse_error_relative_%s_%.2e.dat", dist_name);
    std::sprintf(fname_t, "radon_angle_integrator_transverse_error_relative_%s_%.2e.dat", dist_name);
    std::ofstream output_nt{};
    output_nt.open(fname_nt);
    std::ofstream output_t{};
    output_t.open(fname_t);
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::array<double, 2> error = {};
            if (reference(i, j)[0] != 0.0 && reference(i, j)[1] != 0.0)
                error = relative_error(out(i, j), reference(i, j));
            output_nt << error[0] << ' ';
            output_t << error[1] << ' ';
        }
        output_nt << '\n';
        output_t << '\n';
    }
    output_nt.close();
    output_t.close();
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, std::span<const double> relerrs, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds)
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
    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_order).transform(
            dist, 1.0, reference_order);
    
    std::vector<std::array<double, 2>> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zebra::IsotropicTransverseAngleIntegrator integrator(reference_order);
    integrator.integrate(reference_distribution, boosts, min_speeds, reference);

    bench.title("integrate::RadonAngleIntegrator::integrate");
    for (double relerr : relerrs)
    {
        char name[32] = {};
        std::sprintf(name, "%.16e", relerr);
        benchmark_radon_angle_integrator_isotropic(
                bench, name, dist, dist_name, boosts, min_speeds, relerr, reference);
    }

    char fname[512] = {};
    std::sprintf(fname, "radon_angle_integrator_isotropic_bench_%s_%.2f_%lu_%lu.json", dist_name, boost_len, num_boosts, num_min_speeds);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}

int main([[maybe_unused]] int argc, char** argv)
{
    const std::size_t bench_ind = atoi(argv[1]);
    const double boost_len = atof(argv[2]);
    const std::size_t num_boosts = atoi(argv[3]);
    const std::size_t num_min_speeds = atoi(argv[4]);

    const std::vector<double> relerrs = {
        1.0e+2, 1.0e+1, 1.0e+0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10
    };

    switch (bench_ind)
    {
        case 0:
        {
            std::span<const double> aniso_gaussian_relerrs(
                    relerrs.data(), relerrs.size());
            run_benchmarks(
                    aniso_gaussian, "aniso_gaussian", aniso_gaussian_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 1:
        {
            std::span<const double> four_gaussians_relerrs(
                    relerrs.data(), relerrs.size());
            run_benchmarks(
                    four_gaussians, "four_gaussians", four_gaussians_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 2:
        {
            std::span<const double> shm_plus_stream_relerrs(
                    relerrs.data(), relerrs.size() - 2);
            run_benchmarks(
                    shm_plus_stream, "shm_plus_stream", shm_plus_stream_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 3:
        {
            std::span<const double> shmpp_aniso_relerrs(
                    relerrs.data(), relerrs.size());
            run_benchmarks(
                    shmpp_aniso, "shmpp_aniso", shmpp_aniso_relerrs, boost_len, 
                    num_boosts, num_min_speeds);
            break;
        }
        case 4:
        {
            std::span<const double> shmpp_relerrs(
                    relerrs.data(), relerrs.size() - 2);
            run_benchmarks(
                    shmpp, "shmpp", shmpp_relerrs, boost_len, num_boosts, 
                    num_min_speeds);
            break;
        }
    }
}