.. _sec-vectors-and_matrices:

Vectors and Matrices
====================

The headers `vector.hpp` and `matrix.hpp` provide a vector type and a number of matrix types, which
are integral for using the library. These types represent small vectors and matrices whose sizes
are known at compile time. These are mainly used to represent three-dimensional vectors such as
velocities, and the transformation (e.g. rotations) applies to those vectors. They come with the
standard overloaded operations you would expect: vector addition, scalar multiplication, and matrix
multiplication, plus other miscellaneous linear algebra operations.

.. _subsec-vectors:

Vectors
-------

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

.. _subsec-matrix-memory-layout:

Matrix memory layout
--------------------

Matrices are two-dimensional objects, but computer memory is one-dimensional. There exist two
common conventions for representing a matrix using a one-dimensional segment of memory. Suppose we
have a matrix

.. math::

    \begin{matrix}
        A_{11} & A_{12} & A_{13} \\
        A_{21} & A_{22} & A_{23} \\
    \end{matrix},

In *row-major* layout the elements would be ordered in memory with each row laid out after another,

.. math::

    A_{11}, A_{12}, A_{13}, A_{21}, A_{22}, A_{23}.

In *column-major* layout the elements are instead odrdered with each column after another,

.. math::

    A_{11}, A_{21}, A_{12}, A_{22}, A_{13}, A_{23}.

This library gives the user control over the matrix layout via the enum
:cpp:enum:`zdm::la::MatrixLayout`. The user can provide a template parameter of this type to each
of the matrix types in this library, though this is not necessary, and by default matrices use
a column-major layout.

.. _subsec-active-and-passive-transformations:

Active and passive transformations
----------------------------------

Linear transformations (in general, geometric transformations) for objects defined in a coordinate
system can be described in two ways
    - active transformations: transform objects (e.g. vectors) in space, leaving the coordinate
      system unchanged.
    - passive transformations: transform the coordinate system, leaving the objects unchanged.

See the `Wikipedia article <https://en.wikipedia.org/wiki/Active_and_passive_transformation>`_ for
a more detailed description.

Due to the ambiguity over whether a given transformation represents an active or passive
transformation, this library tags each transformation type with a template parameter of type
:cpp:enum:`zdm::la::Action`, which has the values ``active`` and ``passive``.

.. _subsec-transformation-chaining:

Transformation chaining
-----------------------

When multiple geometric transformations are chained, call them :math:`T_1,\ldots,T_N`, there are
two conventional coordinate system choices for writing them:
    - extrinsic: all transformations are written in the original coordinate system.
    - intrinsic: all transformations are written in a new transformed cooordinate system
      determined by the transformation.

This issue most commonly arises in describing three-dimensional rotations in terms of `Euler angles
<https://en.wikipedia.org/wiki/Euler_angles>`_. Typically, the extrinsic convention is implicitly
used for active transformations, while the intrinsic convention is implicitly used for passive
transformations.

To make this choice of convention explicit, this library compels the user to provide a
:cpp:enum:`zdm::la::Chaining` tag as a template argument to functions where this issue of
composition order is relevant.

.. _subsec-general-matrices:

General Matrices
----------------

.. doxygenstruct:: zdm::la::Matrix
    :project: zebradm
    :no-link:
    :outline:

This type represents a general :math:`N\times M`-matrix. Like :cpp:type:`zdm::la::Vector`, it takes
an arithmetic value type. It takes two values for the dimensions of the matrix. In addition, it
has the template parameters: ``action_param`` of type :cpp:enum:`zdm::la::Action`, and
``layout_param`` of type :cpp:enum:`zdm::la::MatrixLayout`.

Matrix multiplication operations are implemented for this matrix type via operator overloads.

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
    const zdm::la::Vector<double, 3> u1 = m1*v;     // matrix-vector multiplication
    const zdm::la::Matrix<double, 3, 3> m3 = m1*m2; // matrix-matrix multiplication

.. _subsec-rotation-matrices:

Rotation Matrices
-----------------

.. doxygenstruct:: zdm::la::RotationMatrix
    :project: zebradm
    :no-link:
    :outline:

This type represents a rotation matrix. It is used extensively by the library when dealing with
rotations. It has the exact same template parameters as the general matrix type.

A :math:`2\times 2` rotation matrix can be constructed from an angle

.. code:: cpp

   const auto r1 = zdm::la::RotationMatrix<double, 2>::from_angle(0.5*std::numbers::pi);

The class offers multiple ways to construct :math:`3\times 3` rotation matrices

.. code:: cpp

    // Rotation about coordinate axis
    const auto r1 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);

    // From axis-angle
    const auto r2 = zdm::la::RotationMatrix<double, 3>::axis(zdm::la::Vector<double, 3>{1.0, 1.0, 1.0}, std::numbers::pi);

    constexpr auto chaining = zdm::la::Chaining::intrinsic;

    // From (proper) Euler angles
    constexpr auto euler_convention = zdm::la::EulerConvention::zyz;
    const auto r3 = zdm::la::RotationMatrix<double, 3>::from_euler_angles<euler_convention, chaining>(
        std::numbers::pi, 0.5*std::numbers::pi, std::numbers::pi);

    // From Tait-Bryan angles
    constexpr auto tait_bryan_convention = zdm::la::EulerConvention::xyz;
    const auto r3 = zdm::la::RotationMatrix<double, 3>::from_tait_bryan_angles<tait_bryan_convention, chaining>(
        std::numbers::pi, 0.5*std::numbers::pi, std::numbers::pi);

    // Matrix that aligns a vector with the z-axis
    const auto r4 = zdm::la::RotationMatrix<double, 3>::align_z(la::Vector<double, 3>{1.0, 2.0, 3.0});

The enums :cpp:enum:`zdm::la::Chaining`, :cpp:enum:`zdm::la::EulerConvention` and
:cpp:enum:`zdm::la::TaitBryanConvention` cover the different conventions for constructing rotation
matrices from three rotations about the coordinate axes. See the section
:ref:`subsec-transformation-chaining` for more details on chaining conventions. The other two conventions
refer to the choices of axes around which the different rotations or. We refer to the Wikipedia
article on `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_ for a detailed description
on the various conventions.

The rotation matrices also implement matrix multiplication operations via operator overloads

.. code:: cpp

    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    const zdm::la::Matrix<double, 3, 3> m1 = {
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };
    const auto r1 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);
    const auto r2 = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::Axis::y>(0.5*std::numbers::pi);

    const zdm::la::Vectot<duoble, 3> u2 = r1*v;                     // rotation-vector multiplication
    const zdm::la::RotationMatrix<double, 3> r3 = r1*r2;            // rotation-rotation multiplication
    const zdm::la::Matrix<double, 3, 3> m4 = r1*m1*r1.inverse();    // rotation-matrix multiplication

.. _subsec-rigid-transforms:

Rigid Transforms
----------------

.. doxygenstruct:: zdm::la::RigidTransform
    :project: zebradm
    :no-link:
    :outline:

This type represents a rigid transformation. A rigid transformation is a combinatioin of a rotation
and a translation. Therefore, a rigid transformation is naturally constructed from a rotation and
a translation

.. code:: cpp

    const auto rotation = zdm::la::RotationMatrix<double, 3>::coordinate_axis<zdm::z>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> translation = {1.0, 2.0, 3.0};

    const auto chaining = zdm::la::Chaining::intrinsic;
    const auto t1 = zdm::la::RigidTransform<double, 3>::from<chaining>(rotation, translation);
    const auto t2 = zdm::la::RigidTransform<double, 3>::from(rotation);
    const auto t3 = zdm::la::RigidTransform<double, 3>::from(translation);

The last two cases correspond to a pure rotation and a pure translation, respectively. The
parameter ``chaining`` here refers to chaining order of the rotation and translation in the
construction of the rigid transform (see section :ref:`subsec-transformation-chaining`). In the
intrinsic convention, ``translation`` is written in the coordinates that are obtained by applying
``rotation`` to the coordinate axes, while in the extrnsic convention it is expressed in the source
coordinate system.

Since rigid transformations do not act as matrices, they do not have a multiplication operator
defined for them. Instead there is a function for composing them

.. code:: cpp

    const zdm::la::RigidTransform<double, 3> t4 = compose<chaining>(t1, t2);

Note that this function acts in the sense of function composition. That is ``t`` describes a rigid
transformation, which first transforms an object by ``t1`` and then by ``t2``. It is also possible
to compose rigid transformations with rotation matrices and translation vectors

.. code:: cpp

    const zdm::la::Vector<double, 3> v = {1.0, 0.0, -1.0};
    const auto r = zdm::la::RotationMatrix<double, 3>::axis(v, std::numbers::pi);

    const zdm::la::RigidTransform<double, 3> t5 = compose<chaining>(t1, v);
    const zdm::la::RigidTransform<double, 3> t6 = compose<chaining>(t5, r);

Rigid transforms act on vectors and matrices via their call operator

.. code:: cpp

    const zdm::la::Vector<double, 3> u = t6(v);

    const zdm::la::Matrix<double, 3, 3> m1 = {
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };

    const zdm::la::Matrix<double, 3, 3> m2 = t6(m1);

The action of a rigid transform on a matrix corresponds to rotating the matrix by the rotation part
of the rigid transform.

Reference
---------
