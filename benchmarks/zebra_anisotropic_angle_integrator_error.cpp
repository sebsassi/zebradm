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

#include "distributions.hpp"
#include "responses.hpp"

void angle_integrator_error(
    std::span<const std::array<double, 3>> boosts, std::span<const double > eras, std::span<const double> min_speeds, zest::MDSpan<double, 2> reference, DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, std::size_t dist_order, std::size_t resp_order, bool relative_error)
{
    zest::zt::ZernikeExpansion distribution
        = zest::zt::ZernikeTransformerOrthoGeo(dist_order).transform(
            dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        min_speeds.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::zebra::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {min_speeds.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, min_speeds, response);

    std::vector<double> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);
    integrator.integrate(distribution, response, boosts, eras, min_speeds, out);

    char fname[512] = {};
    if (relative_error)
        std::sprintf(fname, "zebra_angle_integrator_error_relative_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    else
        std::sprintf(fname, "zebra_angle_integrator_error_absolute_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    std::ofstream output{};
    output.open(fname);
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

void fill_reference(
    DistributionSpherical dist, Response resp, std::span<const std::array<double, 3>> boosts, std::span<const double > eras, std::span<const double> min_speeds, zest::MDSpan<double, 2> reference)
{
    std::printf("Initalizing reference... ");
    std::fflush(stdout);
    constexpr std::size_t reference_dist_order = 200;
    constexpr std::size_t reference_resp_order = 800;

    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_dist_order).transform(
            dist, 1.0, reference_dist_order);
    
    std::vector<std::array<double, 2>> reference_response_buffer(
        min_speeds.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(reference_resp_order));
    zdm::zebra::SHExpansionVectorSpan<std::array<double, 2>>
    reference_response(reference_response_buffer.data(), {min_speeds.size()}, reference_resp_order);
    zdm::zebra::ResponseTransformer(reference_resp_order).transform(resp, min_speeds, reference_response);
    
    zdm::zebra::AnisotropicAngleIntegrator integrator(reference_dist_order, reference_resp_order);
    integrator.integrate(reference_distribution, reference_response, boosts, eras, min_speeds, reference);
    std::printf("Done\n");
}

void angle_integrator_errors(
    DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, bool relative_error, double boost_len, std::size_t num_boosts, std::size_t num_min_speeds)
{
    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> boosts(num_boosts);
    std::vector<double> eras(num_boosts);
    for (std::size_t i = 0; i < num_boosts; ++i)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        boosts[i] = {
            boost_len*st*std::cos(az), boost_len*st*std::sin(az), ct
        };
        eras[i] = 2.0*std::numbers::pi*rng_dist(gen);
    }

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*(boost_len + 1.0)/double(num_min_speeds - 1);

    std::vector<double> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});

    fill_reference(dist, resp, boosts, eras, min_speeds, reference);

    const std::vector<std::size_t> dist_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};
    const std::vector<std::size_t> resp_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,400,480};

    for (std::size_t dist_order : dist_orders)
    {
        for (std::size_t resp_order : resp_orders)
        {
            std::printf("%lu %lu\n", dist_order, resp_order);
            angle_integrator_error(
                    boosts, eras, min_speeds, reference, dist, dist_name, resp, resp_name, dist_order, resp_order, relative_error);
        }
    }
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr bool relative_error = true;
    constexpr std::array<Labeled<DistributionSpherical>, 5> distributions = {
        Labeled<DistributionSpherical>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionSpherical>{four_gaussians, "four_gaussians"},
        Labeled<DistributionSpherical>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionSpherical>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionSpherical>{shmpp, "shmpp"}
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

    const Labeled<DistributionSpherical> dist = distributions[dist_ind];
    const Labeled<Response> resp = responses[resp_ind];
    std::printf("%s %s\n", dist.label, resp.label);

    angle_integrator_errors(
        dist.object, dist.label, resp.object, resp.label, relative_error, 
        boost_len, num_boosts, num_min_speeds);
}