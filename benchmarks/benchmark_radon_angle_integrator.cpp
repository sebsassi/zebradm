#include <random>
#include <fstream>

#include "radon_integrator.hpp"
#include "nanobench.h"
#include "distributions.hpp"

void benchmark_radon_angle_integrator_isotropic(
    ankerl::nanobench::Bench& bench, const char* name, DistributionCartesian dist, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds, double relerr)
{
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

    std::vector<double> out_buffer(num_boosts*num_min_speeds);
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    std::size_t max_subdiv = 200000000;
    integrate::RadonAngleIntegrator integrator{};
    bench.run(name, [&](){
        integrator.integrate(
                dist, boosts, min_speeds, 0.0, relerr, out, max_subdiv);
    });
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, std::span<const double> relerrs, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));
    bench.maxEpochTime(std::chrono::nanoseconds(100000000000));

    bench.title("integrate::RadonAngleIntegrator::integrate");
    for (double relerr : relerrs)
    {
        char name[32] = {};
        std::sprintf(name, "%.16e", relerr);
        benchmark_radon_angle_integrator_isotropic(
                bench, name, dist, boost_len, num_boosts, num_min_speeds, relerr);
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

    switch (bench_ind)
    {
        case 0:
        {
            std::vector<double> aniso_gaussian_relerrs = {
                1.914959000000000300e+00,
                9.835629000000004063e-02,
                9.095030000000001169e-02,
                9.677714000000000455e-03,
                9.683745000000005751e-03,
                9.059205000000005500e-04,
                9.052364000000004797e-04,
                6.898901000000001390e-05,
                6.916474000000000146e-05,
                5.233195000000000639e-06,
                3.492654000000000988e-07,
                1.919695000000000369e-08,
                1.327729000000000224e-09,
                6.068014000000003309e-11,
            };
            run_benchmarks(
                    aniso_gaussian, "aniso_gaussian", aniso_gaussian_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 1:
        {
            std::vector<double> four_gaussians_relerrs = {
                7.155964000000005854e+01,
                1.351470999999999911e+01,
                1.044743000000000066e+01,
                1.002399000000001017e+01,
                6.417217000000014160e+00,
                6.760092000000001100e+00,
                3.939549000000000856e+00,
                3.270605000000004203e+00,
                3.254509000000005425e+00,
                2.563004000000002058e+00,
                1.621836000000000055e+00,
                1.135130000000001749e+00,
                8.536006000000005978e-01,
                5.620267000000006563e-01,
                2.792287000000000519e-01,
                1.661539000000000765e-01,
                1.313647000000001397e-01,
                8.517672000000005295e-02,
                3.764950000000007180e-02,
                1.569492000000003926e-02,
                6.098417000000002676e-03,
                2.129554000000000082e-03,
                6.357878000000009942e-04,
                1.787007000000002727e-04,
                1.121172000000000452e-05,
                4.026271000000000677e-07,
                9.871452000000002981e-09,
                1.711743000000000155e-10
            };
            run_benchmarks(
                    four_gaussians, "four_gaussians", four_gaussians_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 2:
        {
            std::vector<double> shm_plus_stream_relerrs = {
                1.489760000000000950e+01,
                1.111051000000000011e+00,
                1.108368000000000020e+00,
                9.886651000000000744e-01,
                1.065397000000000149e+00,
                1.656076000000000215e-01,
                1.671954000000000495e-01,
                4.438776000000002608e-02,
                4.478651000000008486e-02,
                8.516546000000003308e-03,
                1.718495000000003475e-03,
                5.236359000000000326e-04,
                2.131566000000003030e-04,
                7.885751000000005672e-05,
                6.871824000000007443e-06,
                3.802178000000002774e-07,
                1.581297000000000438e-08
            };
            run_benchmarks(
                    shm_plus_stream, "shm_plus_stream", shm_plus_stream_relerrs, 
                    boost_len, num_boosts, num_min_speeds);
            break;
        }
        case 3:
        {
            std::vector<double> shmpp_aniso_relerrs = {
                4.002491000000009649e+02,
                4.233904000000000423e+01,
                5.097143000000003354e+01,
                2.970017000000001417e+01,
                2.939803000000006961e+01,
                1.935387000000002544e+01,
                1.931074000000000623e+01,
                1.017631000000000263e+01,
                1.017770000000001573e+01,
                5.174211000000018323e+00,
                3.314788000000007173e+00,
                2.149495000000000378e+00,
                1.249081000000001662e+00,
                6.376409000000012872e-01,
                1.095586000000000060e-01,
                2.818860000000009483e-02,
                3.109770000000014294e-03,
                6.439943000000001574e-04,
                6.804880000000002685e-06,
                4.131581000000001638e-08,
                1.286453000000003012e-10
            };
            run_benchmarks(
                    shmpp_aniso, "shmpp_aniso", shmpp_aniso_relerrs, boost_len, 
                    num_boosts, num_min_speeds);
            break;
        }
        case 4:
        {
            std::vector<double> shmpp_relerrs = {
                1.939205000000003665e+01,
                1.233359000000000094e+00,
                1.283836000000000199e+00,
                1.463042000000001064e+00,
                1.459563999999999862e+00,
                2.564345000000000652e-01,
                2.556614000000000386e-01,
                9.727971000000014412e-02,
                9.831314000000002107e-02,
                3.766056000000002346e-02,
                1.800946000000000491e-02,
                8.900124000000007515e-03,
                5.002145000000006447e-03,
                2.997985000000000753e-03,
                4.691835000000000320e-04,
                1.269662999999999977e-04,
                1.382115000000000403e-05,
                2.761087000000002098e-06,
                3.021453000000000526e-08
            };
            run_benchmarks(
                    shmpp, "shmpp", shmpp_relerrs, boost_len, num_boosts, 
                    num_min_speeds);
            break;
        }
    }
}