Library reference
=================

Zernike-based Radon transforms
------------------------------

Types
^^^^^

.. doxygenclass:: zdm::zebra::IsotropicAngleIntegrator
    :project: zebradm
    :members:

.. doxygenclass:: zdm::zebra::AnisotropicAngleIntegrator
    :project: zebradm
    :members:

.. doxygenclass:: zdm::zebra::IsotropicTransverseAngleIntegrator
    :project: zebradm
    :members:

.. doxygenclass:: zdm::zebra::AnisotropicTransverseAngleIntegrator
    :project: zebradm
    :members:

Functions
^^^^^^^^^

.. doxygenfunction:: zdm::zebra::radon_transform
    :project: zebradm

.. doxygenfunction:: zdm::zebra::radon_transform_inplace
    :project: zebradm

Containers and views
--------------------

Types
^^^^^

.. doxygenclass:: zdm::SHExpansionVector
    :project: zebradm
    :members:

.. doxygenclass:: zdm::SuperSpan
    :project: zebradm
    :members:

.. doxygenclass:: zdm::MultiSuperSpan
    :project: zebradm
    :members:

Type aliases
^^^^^^^^^^^^

.. doxygentypedef:: zdm::SHExpansion
    :project: zebradm

.. doxygentypedef:: zdm::ZernikeExpansion
    :project: zebradm

.. doxygentypedef:: zdm::SHExpansionSpan
    :project: zebradm

.. doxygentypedef:: zdm::SHExpansionVectorSpan
    :project: zebradm

.. doxygentypedef:: zdm::ZernikeExpansionSpan
    :project: zebradm

Coordinate transforms
---------------------

Functions
^^^^^^^^^

.. doxygenfunction:: zdm::coordinates::cartesian_to_spherical_geo
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_geo(double, double)
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_geo(double, double, double)
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::cartesian_to_spherical_phys
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_phys(double, double)
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_phys(double, double, double)
    :project: zebradm

Miscellaneous
-------------

Types
^^^^^

.. doxygenclass:: zdm::zebra::ResponseTransformer
    :project: zebradm
    :members:
