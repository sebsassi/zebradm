/*
Copyright (c) 2025 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#pragma once

namespace zdm::la
{

/**
    @brief Enum specifying a geomtric transformation convention.

    Transformations can either transform the target object (e.g. a vector), or
    they can transform the coordinate system in which the target object is
    defined. In the former case, the transformation is called active, while
    in the latter case it is called passive.
*/
enum class Action
{
    active,
    passive
};

/**
    @brief Enum specifying a geometric transformation composition convention.

    When multiple transformations are composed, there exists an ambiquity over
    in which coordinate system the successive transformation are defined. That
    is, given two transformations \f$T_1\f$ and \f$T_2\f$, is \f$T_2\f$ defined
    relative to the original coordinate system (extrinsic convention), or is it
    defined relative to the intermediate coordinate system determined by
    \f$T_1\f$ (intrinsic convention)?

    This choice is typically implicit, usually made such that active
    transformations are implicitly composed in the extrinsic convention, and
    passive transformations are composed in the intrinsic convention.
*/
enum class Chaining
{
    intrinsic,
    extrinsic
};

} // namespace zdm::la
