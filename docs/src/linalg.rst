Vectors and Matrices
====================

The headers `vector.hpp` and `matrix.hpp` provide a vector type and a number of matrix types, which
are integral for using the library. These types represent small vectors and matrices whose sizes
are known at compile time. These are mainly used to represent three-dimensional vectors such as
velocities, and the transformation (e.g. rotations) applies to those vectors. They come with the
standard overloaded operations you would expect: vector addition, scalar multiplication, and matrix
multiplication, plus other miscellaneous linear algebra operations.

.. doxygenstruct:: zdm::la::Vector
    :project: zebradm
    :no-link:
    :outline:

This is the basic vector type of the library. It is a template type which takes as arguments the
value type of the elements of the vector, and the size of the vector. This type is very similar to
the standard library type :cpp:type:`std::array`. In fact, it is completely interconvertible with
:cpp:type:`std::array` and replicates its interface. The only difference is that it requires the
value type to be an arithmetic type (in practice, either a floating point or an integral type),
and it has operator overloads for common mathematical operations

.. code:: cpp

   const double a = 2.0;
   const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
   const zdm::la::Vector<double, 3> u = {1.0, 0.0, 1.0};

   const zdm::la::Vector<double, 3> w = a*v + u;

As mentioned above, there are a number of other functions you would expect for vectors

.. code:: cpp

    const double dot = zdm::la::dot(v, u);
    const double c = zdm::la::length(v);

    const zdm::la::Vector<double, 3> cross = zdm::la::cross(v, u);
    const zdm::la::Vector<double, 3> normed = zdm::la::normalize(v);

Two basic matrix types are available in this library.

.. doxygenstruct:: zdm::la::Matrix
    :project: zebradm
    :no-link:
    :outline:

This type represents a general :math:`N\times M`-matrix. Like :cpp:type:`zdm::la::Vector`, it takes
an arithmetic value type. It takes two values for the dimensions of the matrix. In addition, it
has two extra template parameters: ``action_param`` of type :cpp:enum:`zdm::la::Action`, and
``layout_param`` of type :cpp:enum:`zdm::la::MatrixLayout`. 

The first parameter exists to make the distinction between `active and passive transformations
<https://en.wikipedia.org/wiki/Active_and_passive_transformation>`_ explicit so that the user can
make it clear, which type of transformation the matrix is intended to represent. The default
transformation type is passive.

The second parameter describes the order in which the elements of the matrix are laid out in
memory. There are two options here: row major (C-style) and column major (Fortran-style) layout.
Given a matrix

.. math::

    \begin{matrix}
        A_{11} & A_{12} & A_{13} \\
        A_{21} & A_{22} & A_{23} \\
    \end{matrix},

in row-major layout the elements would be ordered in memory as

.. math::

    A_{11}, A_{12}, A_{13}, A_{21}, A_{22}, A_{23}.

In column-major layout they would be ordered as

.. math::

    A_{11}, A_{21}, A_{12}, A_{22}, A_{13}, A_{23}.

During use, this is primarily relevant when initializing a matrix from a list of elements. This
library defaults to column-major layout.

.. doxygenstruct:: zdm::la::RotationMatrix
    :project: zebradm
    :no-link:
    :outline:

This is the second matrix type, representing a rotation matrix. It is used extensively by the
library when dealing with rotations. It has the exact same template parameters as the general
matrix type. The class offers multiple ways to construct :math:`3\times 3` rotation matrices

.. code:: cpp

    // Rotation about one axis
    const auto r1 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);

    // Rotation from axis-angle
    const auto r2 = zdm::la::RotationMatrix<double, 3>::axis(zdm::la::Vector<double, 3>{1.0, 1.0, 1.0}, std::numbers::pi);

    // Euler angles
    constexpr auto convention = zdm::la::EulerConvention::zyz;
    constexpr auto chaining = zdm::la::Chaining::intrinsic;
    const auto r2 = zdm::la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
        std::numbers::pi, 0.5*std::numbers::pi, std::numbers::pi);

    // And more... See reference for all possibilities.

Both types of matrices implement the expected operations

.. code:: cpp

    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    const zdm::la::Matrix<double, 3, 3> m1 = {
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };
    const zdm::la::Matrix<double, 3, 3> m2 = {
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0
    };
    const auto r1 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);
    const auto r2 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::y>(0.5*std::numbers::pi);

    const zdm::la::Vector<double, 3> u1 = m1*v;
    const zdm::la::Vectot<duoble, 3> u2 = r1*v;

    const zdm::la::Matrix<double, 3, 3> m3 = m1*m2;
    const zdm::la::RotationMatrix<double, 3> r3 = r1*r2;

    const zdm::la::Matrix<double, 3, 3> m4 = r1*m1*r1.inverse();

