#include "radon_integrator.hpp"
#include "zebra_angle_integrator.hpp"

double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

template <typename FuncType>
void test_mutual_convergence(FuncType&& f)
{
    const std::vector<Vector<double, 3>> boosts = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    const std::vector<double> min_speeds = {
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    std::vector<double> integrator_test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> integrator_test(
            integrator_test_buffer.data(), {boosts.size(), min_speeds.size()});

    integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(f, boosts, min_speeds, 0.0, 1.0e-9, integrator_test);

    std::vector<double> transformer_test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> transformer_test(
            transformer_test_buffer.data(), {boosts.size(), min_speeds.size()});

    constexpr std::size_t order = 200;
    zest::zt::ZernikeExpansionOrthoGeo distribution
        = zest::zt::ZernikeTransformerOrthoGeo<>(order).transform(
                f, 1.0, order);

    zebra::IsotropicAngleIntegrator(order).integrate(
            distribution, boosts, min_speeds, transformer_test);

    std::printf("integrator\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", integrator_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntransformer\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", transformer_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < boosts.size(); ++i)
    {
        for (std::size_t j = 0; j < min_speeds.size(); ++j)
        {
            std::printf("%.16e ", 1.0 - integrator_test(i, j)/transformer_test(i, j));
        }
        std::printf("\n");
    }
    std::printf("\n");
}

int main()
{
    test_mutual_convergence([](const Vector<double, 3>& v){
        constexpr double disp = 0.4;
        const double speed = length(v);
        const double ratio = speed/disp;
        return std::exp(-ratio*ratio);
    });

    test_mutual_convergence([](const Vector<double, 3>& v)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        return std::exp(-0.5*quadratic_form(sigma, v));
    });
}