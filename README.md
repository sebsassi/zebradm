[![arXiv](https://img.shields.io/badge/arXiv-2504.19714-green?labelColor=red)](https://arxiv.org/abs/2504.19714)

# ZebraDM - fast Zernike based Radon transforms for DM event rate computations

ZebraDM is a modern C++ library that provides tools for performing dark matter direct detection
event rate computations using the fast Zernike based Radon transform introduced in
[arXiv:2504.19714](https://arxiv.org/abs/2504.19714).

**This library is a work in progress.**

## Theoretical background

Dark matter double differential event rate integrals can be expressed as Radon transforms of the
dark matter velocity distribution. Expanding the velocity distribution in a suitable chosen basis
of functions (the 3D Zernike function basis) for which an analytic result for the Radon transform
is available, the problem of computing event rates can be reduced from integration to summation of
expansion coefficients. This has numerous advantages: fast convergence of the basis expansion,
fewer function evaluations, array operations consisting of easy-to-vectorize branchless loops of
addition and multiplication, and so on. The theoretical background is further described in
[arXiv:2504.19714](https://arxiv.org/abs/2504.19714).

## Status and roadmap

This library is in active development. Below is a list of features the library will have sorted
by their status.

Completed features
- Functionality for performing the fast Zernike-based Radon transform described in 
[arXiv:2504.19714](https://arxiv.org/abs/2504.19714), and for evaluating the angle-integrated
response-weighted Radon transform for different combinations of isotropic/anisotropic responses
and distributions.
- Tools for computing the laboratory velocity with respect to the dark matter Halo in various
Earth-based frames.

Features currently being implemented
- DM-electron scattering rate computations.
- DM-nucleon scattering rate in the non-relativistic effective theory.

Future features
- Migdal rate in DM-nucleon scattering.
- DM-electron scattering in non-relativistic effective theory.

## Build and installation

This library depends on two other libraries:
- [zest](https://github.com/sebsassi/zest) is a companion library, which provides
utilities for performing Zernike and spherical harmonic transforms.
- [cubage](https://github.com/sebsassi/cubage) provides capabilities for
multidimensional numerical integration. This library is only used by the numerical integration
implementation of the Radon transforms, which are implemented for comparison.

These dependencies are automatically installed by CMake when needed. The library `cubage` is
installed only if either benchmaks or tests are built.

This library uses CMake for its build/install process. The following commands configure,
build, and install the project to your preferred install directory
```bash
cmake --preset=default
cmake --build build
cmake --install build --prefix <install directory>
```

*Note: this library uses the C++23 standard, and requires a sufficiently modern compiler with
support for the features used by the library. Since this library is a work on progress, the
minimum requirements for the compiler will evolve with the needs of the library and increasing
availability of newer compiler versions.*

*This library is developed on and for Linux platforms. If you're on Windows and need to use MSVC,
>>>>>>> dev
you're on your own.*

## Usage

Below is a short program that calculates the angle-integrated Radon transform (and transverse Radon
transform) for an anisotropic velocity distribution, assuming an isotropic target response.
```cpp
//transverse_radon_example.cpp
#include <vector>
#include <print>

#include "zest/zernike_glq_transformer.hpp"
#include "zebradm/zebra_angle_integrator.hpp"

int main()
{
    auto shm_dist = [](const std::array<double, 3>& v){
        constexpr double disp_sq = 0.4*0.4;
        const double speed_sq = zdm::la::dot(v,v);
        return std::exp(-speed_sq/disp_sq);
    };

    constexpr std::size_t order = 20;
    constexpr double vmax = 1.0;
    zest::zt::ZernikeExpansion dist_expansion
        = zest::zt::ZernikeTransformerNormalGeo{}.forward_transform(shm_dist, vmax, order);

    std::vector<zdm::la::Vector<double, 3>> vlab = {
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> vmin = {0.2, 0.3, 0.4};

    zest::DynamicMDArray<std::array<double, 2>, 2> out{vlab.size(), vmin.size()};

    zdm::zebra::TransverseAngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso>(order)
        .integrate(dist_expansion, vlab, vmin, out);

    for (std::size_t i = 0; i < vlab.size(); ++i)
    {
        for (std::size_t j = 0; j < vmin.size(); ++j)
        {
            const double nontransverse = out[i, j][0];
            const double transverse = out[i, j][1];
            std::print("{} {} ", nontransverse, transverse);
        }
        std::println("");
    }
}
```
After installation of the library, the above code can be compiled with, e.g. GCC,
```
g++ -O3 -std=c++23 -o transverse_radon_example.cpp transverse_radon_example.cpp -lzebra -lzest
```
Note the `-std=c++23` needed to enable the C++23 features required by the library.

More examples of usage are found in the `examples` directory.

<!-- ## Documentation -->
<!---->
<!-- HTML and PDF documentation are available in the `docs` directory. -->
