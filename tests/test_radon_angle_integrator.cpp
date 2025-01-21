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
#include "radon_integrator.hpp"

double angle_integrated_radon_shm(
    const std::array<double, 3>& boost, double min_speed, double disp_speed)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    const double boost_speed = zdm::length(boost);
    const double erf_part
        = std::erf(std::min(1.0,min_speed + boost_speed)/disp_speed)
        - std::erf((min_speed - boost_speed)/disp_speed);
    
    const double exp_prefactor
        = (1.0 + boost_speed - std::max(1.0 - boost_speed, min_speed));
    
    const double inv_disp = 1.0/disp_speed;
    const double prefactor = std::numbers::pi*disp_speed*disp_speed/boost_speed;

    return (2.0*std::numbers::pi)*prefactor*(0.5*sqrt_pi*disp_speed*erf_part - exp_prefactor*std::exp(-inv_disp*inv_disp));
}

void test_radon_integrator_is_accurate_for_shm()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const std::array<double, 3>& velocity){
        const double speed = zdm::length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    std::vector<std::array<double, 3>> boosts = {
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

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(shm_dist, boosts, min_speeds, 0.0, 1.0e-9, shm_test);

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
    std::printf("\n");
}

void test_radon_integrator_resp_is_accurate_for_shm()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const std::array<double, 3>& velocity){
        const double speed = zdm::length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    auto resp = [&](
        [[maybe_unused]] double min_speed, [[maybe_unused]] double azimuth, [[maybe_unused]] double colatitude)
    {
        return 1.0;
    };
    const std::vector<std::array<double, 3>> boosts = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };
    const std::vector<double> eras = {
        0.0, std::numbers::pi, 0.5*std::numbers::pi, 0.0, std::numbers::pi, 0.5*std::numbers::pi
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

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            shm_dist, resp, boosts, eras, min_speeds, 0.0, 1.0e-9, shm_test);

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
    std::printf("\n");
}

int main()
{
    test_radon_integrator_is_accurate_for_shm();
    test_radon_integrator_resp_is_accurate_for_shm();
}