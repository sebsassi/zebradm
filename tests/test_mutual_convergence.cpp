#include "radon_integrator.hpp"
#include "radon_transformer.hpp"

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

    RadonIntegrator integrator{};
    integrator.integrate(f, boosts, min_speeds, 0.0, 1.0e-9, integrator_test);

    std::vector<double> transformer_test_buffer(boosts.size()*min_speeds.size());
    zest::MDSpan<double, 2> transformer_test(
            transformer_test_buffer.data(), {boosts.size(), min_speeds.size()});

    constexpr std::size_t order = 200;
    zest::zt::ZernikeExpansionOrthoGeo distribution
        = zest::zt::ZernikeTransformerOrthoGeo<>(order).transform(
                f, 1.0, order);

    RadonTransformer{}.angle_integrated_transform(
            distribution, boosts, min_speeds, 1000, transformer_test);

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
}

int main()
{
    constexpr double disp = 0.4;
    test_mutual_convergence([&](const Vector<double, 3>& v){
        const double speed = length(v);
        const double ratio = speed/disp;
        return std::exp(-ratio*ratio);
    });
}