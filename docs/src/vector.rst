.. _sec-vectors:

Vectors
=======

:cpp:`#include <zebradm/vector.hpp>`

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

Reference
---------


