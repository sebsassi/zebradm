Getting started
===============

Installation
------------

Before we can start with the installation of the library, we need to take care of its dependencies.
There are two:

- [zest](https://github.com/sebsassi/zest) (REQUIRED) is a companion library, which provides
  utilities for performing Zernike and spherical harmonic transforms.
- [cubage](https://github.com/sebsassi/cubage) (OPTIONAL) provides capabilities for
  multidimensional numerical integration. This library is only used by the numerical integration
  implementation of the Radon transforms, which are implemented for comparison.

For all practical purposes, zest is the only dependency you need to care about. Its installation
process is straightforward and similar to this library.

After you have installed zest, installation of this library proceeds similarly. To obtain the
source, clone the repository

.. code:: console

    git clone https://github.com/sebsassi/zebradm.git
    cd zebradm

If you are familiar with CMake, ZebraDM follows a conventional CMake build/install procedure. Even
if not, the process is straightforward. First, build the project

.. code:: console

    cmake --preset=default
    cmake --build build

The default configuration here should be adequate. After that you can install the built library
from the build directory to your desired location

.. code:: console

    cmake --install build --prefix <install directory>

Here ``install directory`` denotes your preferred installation location.

Basic Usage
-----------

To test the installation and take our first steps in using the library, we can create a short
program that evaluates the isotropic angle integrated transverse and nontransverse Radon transform
for a distribution. To do this,  crate a file ``radon.cpp`` with the contents

.. code:: cpp

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

Now, to compile the code, we use GCC in this example and link our code with ZebraDM

.. code:: console

    g++ -std=c++20 -O3 -march=native -o radon radon.cpp -lzebradm -lzest
    
There are a few things of note here. First, zest is built on the C++20 standard, and therefore
requires a sufficiently modern compiler, which implements the necessary C++20 features. To tell GCC
we are using C++20, we give the flag ``std=c++20``.

Secondly, apart from linking with this library, don't forget to link with the dependencies. In this
case, zest.

Finally, the performance of the library is sensitive to compiler optimizations. As a baseline, we
use the optimization level ``-O3`` to enable all architecture-independent optimizations in GCC. On
top of that, this example enables architecture specific optimizations with the ``-march=native flag``.
This is generally advisable if your code will be running on the same machine it is built on.
However, the situation is different if you expect to be running the same executable on machines
with potentially different architectures. For typical x86, fused multiply-add operations ``-mfma``
and AVX2 SIMD operations ``-mavx2``, should be available on most hardware and are sufficient for
near optimal performance.
