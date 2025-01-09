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

constexpr std::array<double, 2> relative_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(1.0 - test[0]/ref[0]),
        std::fabs(1.0 - test[1]/ref[1])
    };
}

void radon_angle_integrator_anisotropic_transverse_error(
    DistributionCartesian dist, const char* dist_name, Response resp, const char* resp_name, const std::span<const std::array<double, 3>> boosts, std::span<const double> eras, std::span<const double> min_speeds, double relerr, std::size_t max_subdiv, zest::MDSpan<const std::array<double, 2>, 2> reference)
{
    std::vector<std::array<double, 2>> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(
            dist, resp, boosts, eras, min_speeds, 0.0, relerr, out, max_subdiv);

    char fname_nt[512] = {};
    char fname_t[512] = {};
    std::sprintf(fname_nt, "radon_anisotropic_angle_integrator_nontransverse_error_relative_%s_%s_%lu_%.2e.dat", dist_name, resp_name, max_subdiv, relerr);
    std::sprintf(fname_t, "radon_anisotropic_angle_integrator_transverse_error_relative_%s_%s_%lu_%.2e.dat", dist_name, resp_name, max_subdiv, relerr);
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
            {
                error = relative_error(out(i, j), reference(i, j));
            }
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
    DistributionCartesian dist, const char* dist_name, Response resp, const char* resp_name, double relerr, std::size_t max_subdiv, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds)
{
    std::printf("Initalizing...\n");
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

    constexpr std::size_t reference_dist_order = 200;
    constexpr std::size_t reference_resp_order = 300;
    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_dist_order).transform(
            dist, 1.0, reference_resp_order);

    std::vector<std::array<double, 2>> response_buffer(
        min_speeds.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(reference_resp_order));
    zdm::zebra::SHExpansionVectorSpan<std::array<double, 2>>
    reference_response(response_buffer.data(), {min_speeds.size()}, reference_resp_order);
    zdm::zebra::ResponseTransformer(reference_resp_order).transform(resp, min_speeds, reference_response);
    
    std::vector<std::array<double, 2>> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zdm::zebra::AnisotropicTransverseAngleIntegrator integrator(reference_dist_order, reference_resp_order);
    integrator.integrate(reference_distribution, reference_response, boosts, eras, min_speeds, reference);
    
    radon_angle_integrator_anisotropic_transverse_error(
            dist, dist_name, resp, resp_name, boosts, eras, min_speeds, relerr, max_subdiv, reference);
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

    const std::size_t dist_ind = atoi(argv[1]);
    const std::size_t resp_ind = atoi(argv[2]);
    const double boost_len = atof(argv[3]);
    const std::size_t num_boosts = atoi(argv[4]);
    const std::size_t num_min_speeds = atoi(argv[5]);
    const double relerr = double(atof(argv[6]));
    const std::size_t max_subdiv = atoi(argv[7]);
    std::printf("dist_ind: %lu\nresp_ind: %lu\nboost_len: %.2f\nnum_boosts: %lu\nnum_speeds: %lu\nrelerr: %.0e\nmax_subdiv: %lu\n", dist_ind, resp_ind, boost_len, num_boosts, num_min_speeds, relerr, max_subdiv);

    const Labeled<DistributionCartesian> dist = distributions[dist_ind];
    const Labeled<Response> resp = responses[resp_ind];

    run_benchmarks(
            dist.object, dist.label, resp.object, resp.label, relerr, max_subdiv, boost_len, num_boosts, num_min_speeds);
}