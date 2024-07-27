#include "radon_transformer.hpp"

#include <cassert>

[[maybe_unused]] constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

[[maybe_unused]] constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::max(std::fabs(a[0] - b[0]), std::fabs(a[1] - b[1])) < tol;
}

bool test_radoon_transformer_is_correct_for_constant_dist()
{
    std::vector<Vector<double, 3>> boosts = {
        {1.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> min_speeds = {0.0, 0.5, 1.0, 1.5};

    std::vector<double> reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {boosts.size(), min_speeds.size()});

    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            constexpr double two_pi_sq = 2.0*std::numbers::pi*std::numbers::pi;
            const double boost_speed = length(boosts[i]);
            const double umax = std::min(1.0, min_speeds[j] + boost_speed);
            const double umin = std::max(-1.0, min_speeds[j] - boost_speed);
            const double umax_cb = umax*umax*umax;
            const double umin_cb = umin*umin*umin;
            reference(i, j) = two_pi_sq*((umax - umin) - (umax_cb - umin_cb)/3.0)/boost_speed;
        }
    }

    constexpr std::size_t order = 1;
    zest::zt::ZernikeExpansionOrthoGeo distribution(order);
    distribution(0,0,0) = {1.0/std::sqrt(3.0), 0.0};

    std::vector<double> test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> test(
            test_buffer.data(), {boosts.size(), min_speeds.size()});

    RadonTransformer{}.angle_integrated_transform(
            distribution, boosts, min_speeds, 1000, test);
    
    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
            success = success && is_close(test(i,j), reference(i,j), tol);
    }

    if (!success)
    {
        std::printf("reference\n");
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
            {
                std::printf("%.16e ", reference(i, j));
            }
            std::printf("\n");
        }

        std::printf("\ntest\n");
        for (std::size_t i = 0; i < boosts.size(); ++i)
        {
            for (std::size_t j = 0; j < min_speeds.size(); ++j)
            {
                std::printf("%.16e ", test(i, j));
            }
            std::printf("\n");
        }
    }

    return success;
}

double angle_integrated_radon_shm(
    const Vector<double, 3>& boost, double min_speed, double disp_speed)
{
    constexpr double sqrt_pi = std::sqrt(std::numbers::pi);
    const double boost_speed = length(boost);
    const double erf_part
        = std::erf(std::min(1.0,min_speed + boost_speed)/disp_speed)
        - std::erf((min_speed - boost_speed)/disp_speed);
    
    const double exp_prefactor
        = (1.0 + boost_speed - std::max(1.0 - boost_speed, min_speed));
    
    const double inv_disp = 1.0/disp_speed;
    const double prefactor = std::numbers::pi*disp_speed*disp_speed/boost_speed;

    return (2.0*std::numbers::pi)*prefactor*(0.5*sqrt_pi*disp_speed*erf_part - exp_prefactor*std::exp(-inv_disp*inv_disp));
}

void test_radon_transformer_is_accurate_for_shm()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const Vector<double, 3>& velocity){
        const double speed = length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    std::vector<Vector<double, 3>> boosts = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    const std::vector<double> min_speeds = {
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    std::vector<double> shm_reference_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> shm_reference(
            shm_reference_buffer.data(), {boosts.size(), min_speeds.size()});

    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            shm_reference(i, j) = angle_integrated_radon_shm(
                    boosts[i], min_speeds[j], disp_speed);
        }
    }

    std::vector<double> shm_test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> shm_test(
            shm_test_buffer.data(), {boosts.size(), min_speeds.size()});

    constexpr std::size_t order = 100;
    zest::zt::ZernikeExpansionOrthoGeo distribution
        = zest::zt::ZernikeTransformerOrthoGeo<>(order).transform(
                shm_dist, 1.0, order);

    RadonTransformer{}.angle_integrated_transform(
            distribution, boosts, min_speeds, 1000, shm_test);

    std::printf("reference\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", shm_reference(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntest\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", shm_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", 1.0 - shm_test(i, j)/shm_reference(i, j));
        }
        std::printf("\n");
    }
}

int main()
{
    assert(test_radoon_transformer_is_correct_for_constant_dist());
    test_radon_transformer_is_accurate_for_shm();
}