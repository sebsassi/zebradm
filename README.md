# ZebraDM - fast Zernike based Radon transforms for DM event rate computations

ZebraDM is a modern C++ library that provides tools for performing dark matter direct detection event rate computations using the fast Zernike based Radon transform introduced in [arXiv:????.????](https://example.com).

## Theoretical background

Dark matter double differential event rate integrals can be expressed as Radon transforms of the dark matter velocity distribution. Expanding the velocity distribution in a suitable chosen basis of functions (the 3D Zernike function basis) for which an analytic result for the Radon transform is available, the problem of computing event rates can be reduced from integration to summation of expansion coefficients. This has numerous advantages: fast convergence of the basis expansion, fewer function evaluations, array operations consisting of easy-to-vectorize branchless loops of addition and multiplication, and so on. The theoretical background is further described in [arXiv:????.????](https://example.com).

## About this library

Currently, this library provides functionality for performing the fast Zernike based Radon transform described in [arXiv:????.????](https://example.com), and a miscellaneous assortment of other basic utilities. This library acts as a reference implementation for these methods.

## Build and installation

This library has dependencies on two other libraries:
- [zest](https://github.com/sebsassi/zest) (REQUIRED) is a companion library, which provides utilities for performing Zernike and spherical harmonic transforms.
- [cubage](https://github.com/sebsassi/cubage) (OPTIONAL) provides capabilities for multidimensional numerical integration. This library is only used by the numerical integration implementation of the Radon transforms, which are implemented for comparison.

This library uses CMake for its build/install process. Therefore, provided you have the required dependencies installed, the following three commands configure, build, and install the project to your preferred install directory
```bash
cmake --preset=default
cmake --build build
cmake --install build --prefix <install directory>
```

Note: this library aims to use the C++20 standard. Therefore, a sufficiently modern compiler is required. At least GGC 13 or Clang 17 is recommended. The library may compile at any point with compilers down to GCC 11 and Clang 14, but no guarantees are made about this.

## Usage

For a general dark matter nuclear scattering problem one often needs both the regular and the transverse Radon transform of the distribution, integrated over the recoil directions. For a basic case with isotropic detector response this is provided by `zebra::IsotropicTransverseAngleIntegrator`:
```cpp
//transverse_radon_example.cpp
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
```
After installation of the library, the above code can be compiled with, e.g.,
```
g++ -O3 -std=c++20 -o transverse_radon_example.cpp transverse_radon_example.cpp -lzebra -lzest
```
Note the `-std=c++20` needed to enable the C++20 features required by the library, unless your compiler defaults to C++20.

If the transverse Radon transform is not needed, the class `zebra::IsotropicAngleIntegrator` is provided. If an anisotropic response to recoils is needed, corresponding anisotropic classes are also provided.