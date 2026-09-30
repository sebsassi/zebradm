#include "moments.hpp"

namespace
{

bool test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment moment)
{
    return zdm::zebra::detail::offset_of(moment) + zdm::zebra::detail::inverse_offset_of(moment) == 2;
}

bool test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::IsoMoment moment)
{
    return zdm::zebra::detail::offset_of(moment) + zdm::zebra::detail::inverse_offset_of(moment) == 2;
}

} // namespace

int main()
{
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::identity));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::x));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::y));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::z));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::r2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::x2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::y2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::z2));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::xy));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::xz));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::Moment::yz));

    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::IsoMoment::identity));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::IsoMoment::linear));
    assert(test_offset_of_plus_inverse_offset_of_equals_two(zdm::zebra::IsoMoment::quadratic));

    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::identity, 0) == 2);
    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::transverse, 0) == 4);
    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::full, 0) == 4);

    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::identity, 10) == 12);
    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::transverse, 10) == 14);
    assert(zdm::zebra::max_radon_order(zdm::zebra::MomentCategory::full, 10) == 14);
}
