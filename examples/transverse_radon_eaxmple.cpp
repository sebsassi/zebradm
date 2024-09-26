#include "zest/zernike_glq_transformer.hpp"
#include "zebradm/zebra_angle_integrator.hpp"

int main()
{
    auto shm_dist = [](const Vector<double, 3>& v){
        constexpr double disp_sq = 0.4*0.4;
        const double speed_sq = dot(v,v);
        return std::exp(-speed_sq/disp_sq);
    }

    constexpr std::size_t order = 20;
    constexpr double vmax = 1.0;
    zest::zt::ZernikeExpansion dist_expansion
        = zest::zt::ZernikeTransformerOrthoGeo{}.transform(shm_dist, vmax, order);
    
    std::vector<std::array<double, 3>> vlab = {
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> vmin = {0.2, 0.3, 0.4};

    std::vector<std::array<double, 2>> out_buffer(vlab.size()*vmin.size());
    zest::MDSpan<std::array<double, 2>, 2> out(
            out_buffer.data(), {vlab.size(), vmin.size()});

    zebra::IsotropicTransverseAngleIntegrator(order)
        .integrate(dist_expansion, vlab, vmin, out);
    
    for (std::size_t i = 0; i < 0; ++i)
    {
        const double nontransverse = out[i][j][0];
        const double transverse = out[i][j][1];
        for (std::size_t j = 0; j < 0; ++j)
            std::printf("{%f, %f} ", nontransverse, transverse);
        std::printf("\n");
    }
}