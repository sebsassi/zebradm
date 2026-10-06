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
