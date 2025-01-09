#include <random>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"

#include "distributions.hpp"

void isotropic_convergence(
    const std::array<double, 3>& boost, std::span<const double> min_speeds, std::span<double> reference, DistributionSpherical dist, const char* name, std::size_t order)
{
    zest::zt::ZernikeExpansion distribution
        = zest::zt::ZernikeTransformerOrthoGeo(order).transform(
            dist, 1.0, order);

    std::vector<double> out(min_speeds.size());

    zdm::zebra::IsotropicAngleIntegrator integrator(order);
    integrator.integrate(distribution, boost, min_speeds, out);

    char fname[512] = {};
    std::sprintf(fname, "zebra_convergence_demo_%s_order_%lu.dat", name, order);
    std::ofstream output{};
    output.open(fname);
    for (std::size_t i = 0; i < min_speeds.size(); ++i)
    {
        const double absolute_error = std::fabs(reference[i] - out[i]);
        output << min_speeds[i] << ' ' << out[i] << ' ' << absolute_error  << '\n';
    }
    output.close();
}

void convergence_demo(
    DistributionSpherical dist, const char* name)
{
    constexpr std::size_t reference_order = 200;

    constexpr std::size_t num_boosts = 100;
    constexpr std::size_t num_min_speeds = 100;

    constexpr double max_min_speed = 1.5;

    constexpr std::array<double, 3> boost = {0.5, 0.0, 0.0};

    std::vector<double> min_speeds(num_min_speeds);
    for (std::size_t i = 0; i < num_min_speeds; ++i)
        min_speeds[i] = double(i)*max_min_speed/double(num_min_speeds - 1);

    zest::zt::ZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerOrthoGeo(reference_order).transform(
            dist, 1.0, reference_order);
    
    std::vector<double> reference(min_speeds.size());
    
    zdm::zebra::IsotropicAngleIntegrator integrator(reference_order);
    integrator.integrate(reference_distribution, boost, min_speeds, reference);

    std::vector<std::size_t> orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};

    for (std::size_t order : orders)
        isotropic_convergence(boost, min_speeds, reference, dist, name, order);
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr std::array<Labeled<DistributionSpherical>, 5> distributions = {
        Labeled<DistributionSpherical>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionSpherical>{four_gaussians, "four_gaussians"},
        Labeled<DistributionSpherical>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionSpherical>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionSpherical>{shmpp, "shmpp"}
    };

    const std::size_t dist_ind = atoi(argv[1]);
    const Labeled<DistributionSpherical> dist = distributions[dist_ind];

    convergence_demo(dist.object, dist.label);
}