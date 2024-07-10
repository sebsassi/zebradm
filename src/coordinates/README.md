# Coordinates

The `coordinates` namespace implements a collection of Galilean transformations between some of the main celestial coordinate systems (horizontal, equatorial, heliocentric, galactic) as well as an interface for building composite transformations of these. In addition, basic functions for transforming between Cartesian and spherical coordinates are provided.

For example, to transform between the heliocentric and equatorial coordinates, one can do
```cpp
HeliocentricToEquatorial hel_to_equ(milkyway_coordinates(), earth_orbit());
```
Here the functions `milkyway_coordinates` and `earth_orbit` provide default parameters for the orientation of the Milky Way in the equatorial coordinate system, and Earth's orbit, respectively. We can get the 4-by-4 affine matrix that encodes the relevant transformation at a given moment in time by calling
```cpp
double days_since_epoch = 100.0;
Matrix<double, 4, 4> affine_hel_to_equ = hel_to_equ.transform(days_since_epoch);
```
We can also extract the 3-by-3 rotation matrix
```cpp
Matrix<double, 3, 3> rotation_hel_to_equ = hel_to_equ.rotation(days_since_epoch);
```
We can construct the inverse of this transformation
```cpp
InverseGalileanTransform<HeliocentricToEquatorial> equ_to_hel{hel_to_equ};
```
From this we can get the matrices as before
```cpp
Matrix<double, 4, 4> affine_equ_to_hel = equ_to_hel.transform(days_since_epoch);
Matrix<double, 3, 3> rotation_equ_to_hel = equ_to_hel.rotation(days_since_epoch);
```

If we now construct another transformation to get from equatorial to horizontal coordinates
```cpp
double latitude = 0.0;
double longitude = 0.0;
EquatorialToHorizontal equ_to_hor(latitude, longitude, earth_rotation());
```
we can create a composite transformation
```cpp
CompositeGalileanTransformation hel_to_hor(hel_to_equ, equ_to_hor);
```

Note that operating with the 4-by-4 affine matrix on a 3D vector $(v_x, v_y, v_z)$ requires use of the augmented vector $(v_x, v_y, v_z, 1)$. To transform multiple vectors with the same transformation, a convenience function is provided
```cpp
std::vector<Vector<double, 3>> vectors = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
std::vector<Vector<double, 3>> times = {0.0, 1.0};
transform(vectors, times, hel_to_hor);
```
Similar convenience functions with the name `rotate` are provided for vectors that are unaffected by the boost (e.g., directions) as well as for matrices.

