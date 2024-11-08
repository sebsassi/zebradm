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
#include "../hypercube_integrator.hpp"

#include <chrono>
#include <iostream>

class Timer
{
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;

public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        stop_time = std::chrono::high_resolution_clock::now();
    }

    double measure()
    {
        return double(std::chrono::duration_cast<std::chrono::microseconds>
        (stop_time - start_time).count());
    }
};

template<typename F>
double benchmark(F&& f)
{
    Timer timer{};
    timer.start();
    std::size_t num_iter = f();
    timer.stop();
    return timer.measure()/double(num_iter);
}

template <std::size_t NDIM>
void benchmark_genz_malik_gaussian(std::size_t num_iter)
{
    //constexpr std::size_t NDIM = 2;
    constexpr double sigma = 1.0;
    std::array<double, NDIM> d{};
    for (auto& element : d)
        element = -0.3;
    auto function = [&](const std::array<double, NDIM>& x)
    {
        const auto z = (1.0/sigma)*x;
        const auto z2 = z*z;
        return std::exp(-0.5*(std::accumulate(z2.begin(), z2.end(), 0.0)));
    };

    constexpr double abserr = 1.0e-7;

    using Integrator = cubage::HypercubeIntegrator<std::array<double, NDIM>, double>;
    using Limits = typename Integrator::Limits;
    using Result = typename Integrator::Result;

    std::array<double, NDIM> a{};
    std::array<double, NDIM> b{};
    for (auto& element : a)
        element = -1.0;
    for (auto& element : b)
        element = 1.0;
    const std::vector<Limits> limits = {
        Limits{a, b}
    };
    std::vector<Result> results(num_iter);

    Integrator integrator{};
    const double base_relerr = 1.0e-40/double(num_iter);

    auto bench = [&](){
        for (std::size_t i = 0; i < num_iter; ++i)
        {
            const double relerr = double(i)*base_relerr;
            results[i] = integrator.integrate(function, limits, abserr, relerr);
        }
        return num_iter;
    };
    
    double time = benchmark(bench);

    std::cout << "Genz-Malik gaussian " << NDIM << "D: " << time << " us/iter\n";
}

int main()
{
    constexpr std::size_t num_iter = 10000;
    benchmark_genz_malik_gaussian<2>(num_iter);
    benchmark_genz_malik_gaussian<3>(num_iter);
    benchmark_genz_malik_gaussian<4>(num_iter);
}