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

constexpr std::array<double, 2> relative_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(1.0 - test[0]/ref[0]),
        std::fabs(1.0 - test[1]/ref[1])
    };
}

constexpr std::array<double, 2> absolute_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(ref[0] - test[0]),
        std::fabs(ref[1] - test[1])
    };
}

void angle_integrator_error(
    std::span<const std::array<double, 3>> boosts, std::span<const double > eras, std::span<const double> min_speeds, zest::MDSpan<const std::array<double, 2>, 2> reference, DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, std::size_t dist_order, std::size_t resp_order, bool use_relative_error)
{
    zest::zt::ZernikeExpansion distribution
        = zest::zt::ZernikeTransformerOrthoGeo(dist_order).transform(
            dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        min_speeds.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::zebra::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {min_speeds.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, min_speeds, response);

    std::vector<std::array<double, 2>> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zdm::zebra::AnisotropicTransverseAngleIntegrator integrator(dist_order, resp_order);
    integrator.integrate(distribution, response, boosts, eras, min_speeds, out);

    char fname_nt[512] = {};
    char fname_t[512] = {};
    if (relative_error)
    {
        std::sprintf(fname_nt, "zebra_angle_integrator_nontransverse_error_relative_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
        std::sprintf(fname_t, "zebra_angle_integrator_transverse_error_relative_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    }
    else
    {
        std::sprintf(fname_nt, "zebra_angle_integrator_nontransverse_error_absolute_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
        std::sprintf(fname_t, "zebra_angle_integrator_transverse_error_absolute_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    }
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
                error = (use_relative_error) ?
                    relative_error(out(i, j), reference(i, j))
                    : absolute_error(out(i, j), reference(i, j));
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

void angle_integrator_errors(
    DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, bool relative_error)
{
    constexpr std::size_t reference_dist_order = 200;
    constexpr std::size_t reference_resp_order = 300;

    constexpr std::size_t num_boosts = 100;
    constexpr std::size_t num_min_speeds = 100;

    constexpr double max_min_speed = 1.5;
    constexpr double max_boost_len = 1.0;

    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> boosts(num_boosts);
    std::vector<double> eras(num_boosts);
    for (std::size_t i = 0; i < num_boosts; ++i)
    {
        const double boost_len = double(i)*max_boost_len/double(num_boosts - 1);
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
        min_speeds[i] = double(i)*max_min_speed/double(num_min_speeds - 1);

    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_dist_order).transform(
            dist, 1.0, reference_dist_order);
    
    std::vector<std::array<double, 2>> reference_response_buffer(
        min_speeds.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(reference_resp_order));
    zdm::zebra::SHExpansionVectorSpan<std::array<double, 2>>
    reference_response(reference_response_buffer.data(), {min_speeds.size()}, reference_resp_order);
    zdm::zebra::ResponseTransformer(reference_resp_order).transform(resp, min_speeds, reference_response);

    std::vector<std::array<double, 2>> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zdm::zebra::AnisotropicTransverseAngleIntegrator integrator(reference_dist_order, reference_resp_order);
    integrator.integrate(reference_distribution, reference_response, boosts, eras, min_speeds, reference);

    const std::vector<std::size_t> dist_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};
    const std::vector<std::size_t> resp_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,240,280};

    for (std::size_t dist_order : dist_orders)
    {
        for (std::size_t resp_order : resp_orders)
            angle_integrator_error(
                    boosts, eras, min_speeds, reference, dist, dist_name, resp, resp_name, dist_order, resp_order, relative_error);
    }
}

int main()
{
    const bool relative_error = true;

    angle_integrator_errors(
        aniso_gaussian, "aniso_gaussian",
        smooth_exponential, "smooth_exponential", relative_error);
    angle_integrator_errors(
        four_gaussians, "four_gaussians",
        smooth_exponential, "smooth_exponential", relative_error);
    angle_integrator_errors(
        shm_plus_stream, "shm_plus_stream",
        smooth_exponential, "smooth_exponential", relative_error);
    angle_integrator_errors(
        shmpp_aniso, "shmpp_aniso",
        smooth_exponential, "smooth_exponential", relative_error);
    angle_integrator_errors(
        shmpp, "shmpp",
        smooth_exponential, "smooth_exponential", relative_error);
    
    angle_integrator_errors(
        aniso_gaussian, "aniso_gaussian",
        fcc_dots, "smooth_dots", relative_error);
    angle_integrator_errors(
        four_gaussians, "four_gaussians",
        fcc_dots, "smooth_dots", relative_error);
    angle_integrator_errors(
        shm_plus_stream, "shm_plus_stream",
        fcc_dots, "smooth_dots", relative_error);
    angle_integrator_errors(
        shmpp_aniso, "shmpp_aniso",
        fcc_dots, "smooth_dots", relative_error);
    angle_integrator_errors(
        shmpp, "shmpp",
        fcc_dots, "smooth_dots", relative_error);
}