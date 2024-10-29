#include <random>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"

#include "distributions.hpp"

void angle_integrator_error(
    std::span<const std::array<double, 3>> boosts, std::span<const double > min_speeds, zest::MDSpan<double, 2> reference, DistributionSpherical dist, const char* name, std::size_t order, bool relative_error)
{
    zest::zt::ZernikeExpansion distribution
        = zest::zt::ZernikeTransformerOrthoGeo(order).transform(
            dist, 1.0, order);

    std::vector<double> out_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {boosts.size(), min_speeds.size()});

    zebra::IsotropicAngleIntegrator integrator(order);
    integrator.integrate(distribution, boosts, min_speeds, out);

    char fname[512] = {};
    if (relative_error)
        std::sprintf(fname, "zebra_angle_integrator_error_relative_%s_order_%lu.dat", name, order);
    else
        std::sprintf(fname, "zebra_angle_integrator_error_absolute_%s_order_%lu.dat", name, order);
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

void angle_integrator_errors(
    DistributionSpherical dist, const char* name, bool relative_error)
{
    constexpr std::size_t reference_order = 200;

    constexpr std::size_t num_boosts = 100;
    constexpr std::size_t num_min_speeds = 100;

    constexpr double max_min_speed = 1.5;
    constexpr double max_boost_len = 1.0;

    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> boosts(num_boosts);
    for (std::size_t i = 0; i < num_boosts; ++i)
    {
        const double boost_len = double(i)*max_boost_len/double(num_boosts - 1);
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        boosts[i] = {
            boost_len*st*std::cos(az), boost_len*st*std::sin(az), ct
        };
    }

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*max_min_speed/double(num_min_speeds - 1);

    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_order).transform(
            dist, 1.0, reference_order);
    
    std::vector<double> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});
    
    zebra::IsotropicAngleIntegrator integrator(reference_order);
    integrator.integrate(reference_distribution, boosts, min_speeds, reference);

    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};

    for (std::size_t order : orders)
        angle_integrator_error(
                boosts, min_speeds, reference, dist, name, order, 
                relative_error);
}

int main()
{
    const bool relative_error = true;
    angle_integrator_errors(aniso_gaussian, "aniso_gaussian", relative_error);
    angle_integrator_errors(four_gaussians, "four_gaussians", relative_error);
    angle_integrator_errors(shm_plus_stream, "shm_plus_stream", relative_error);
    angle_integrator_errors(shmpp_aniso, "shmpp_aniso", relative_error);
    angle_integrator_errors(shmpp, "shmpp", relative_error);
}