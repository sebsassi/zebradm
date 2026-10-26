/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include "moments.hpp"

namespace
{

bool test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment moment)
{
    return zdm::detail::offset_of(moment) + zdm::detail::inverse_offset_of(moment) == 2;
}

bool test_offset_of_plus_inverse_offset_of_equals_two(zdm::IsoMoment moment)
{
    return zdm::detail::offset_of(moment) + zdm::detail::inverse_offset_of(moment) == 2;
}

} // namespace

int main()
{
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::identity));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::x));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::y));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::z));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::r2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::x2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::y2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::z2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::xy));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::xz));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::Moment::yz));

    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::IsoMoment::identity));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::IsoMoment::linear));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::IsoMoment::quadratic));

    assert(zdm::max_radon_order(zdm::MomentSet::identity, 0) == 2);
    assert(zdm::max_radon_order(zdm::MomentSet::transverse, 0) == 4);
    assert(zdm::max_radon_order(zdm::MomentSet::full, 0) == 4);

    assert(zdm::max_radon_order(zdm::MomentSet::identity, 10) == 12);
    assert(zdm::max_radon_order(zdm::MomentSet::transverse, 10) == 14);
    assert(zdm::max_radon_order(zdm::MomentSet::full, 10) == 14);
}
