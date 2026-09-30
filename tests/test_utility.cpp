#include "utility.hpp"

namespace
{

bool empty_spans_have_no_overlap()
{
    return !zdm::util::have_overlap(std::span<int>{}, std::span<int>{});
}

bool nonempty_span_overlaps_with_itself()
{
    std::array<int, 39> arr = {};
    std::span<int> span{arr.data(), 5};
    return zdm::util::have_overlap(span, span);
}

bool spans_of_distinct_arrays_have_no_overlap()
{
    std::array<int, 39> arr1 = {};
    std::array<int, 23> arr2 = {};
    std::span<int> span1{arr1.data(), 5};
    std::span<int> span2{arr2.data(), 5};

    return zdm::util::have_overlap(span1, span2);
}

bool inplace_product_of_array_with_zero_array_is_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr2 = {};
    zdm::util::mul(std::span<int>{arr1}, std::span<const int>{arr2});
    return arr1 == arr2;
}

bool inplace_product_of_array_with_identity_array_does_not_change_value()
{

    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr2 = {};
    arr2.fill(1);

    std::array<int, 39> reference_array = arr1;
    zdm::util::mul(std::span<int>{arr1}, std::span<const int>{arr2});
    return arr1 == reference_array;

}

bool inplace_product_of_array_with_zero_is_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    zdm::util::mul(std::span<int>{arr1}, 0);
    return arr1 == std::array<int, 39>{};
}

bool inplace_product_of_array_with_identity_does_not_change_value()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);

    std::array<int, 39> reference_array = arr1;
    zdm::util::mul(std::span<int>{arr1}, 1);
    return arr1 == reference_array;
}

bool product_of_array_with_zero_array_is_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr2 = {};
    std::array<int, 39> res = {};
    zdm::util::mul(std::span<int>{res}, std::span<const int>{arr1}, std::span<const int>{arr2});
    return res == arr2;
}

bool product_of_array_with_identity_array_does_not_change_value()
{

    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr2 = {};
    arr2.fill(1);
    std::array<int, 39> res = {};

    zdm::util::mul(std::span<int>{res}, std::span<const int>{arr1}, std::span<const int>{arr2});
    return res == arr1;

}

bool product_of_array_with_zero_is_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> resr = {};
    zdm::util::mul(std::span<int>{resr}, std::span<const int>{arr1}, 0);
    std::array<int, 39> resl = {};
    zdm::util::mul(std::span<int>{resl}, 0, std::span<const int>{arr1});
    return resr == std::array<int, 39>{} && resl == resr;
}

bool product_of_array_with_identity_does_not_change_value()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> resr = {};
    zdm::util::mul(std::span<int>{resr}, std::span<const int>{arr1}, 1);
    std::array<int, 39> resl = {};
    zdm::util::mul(std::span<int>{resl}, 1, std::span<const int>{arr1});
    return resr == arr1 && resl == resr;
}

bool inplace_fmadd_with_negation_gives_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr2 = {};
    arr1.fill(-1);
    std::array<int, 39> arr3 = {};
    arr1.fill(2);

    zdm::util::fmadd(std::span<int>{arr1}, std::span<const int>{arr2}, std::span<const int>{arr3});
    return arr1 == std::array<int, 39>{};
}

bool inplace_fmadd_with_scalar_negation_gives_zero()
{
    std::array<int, 39> arr1 = {};
    arr1.fill(2);
    std::array<int, 39> arr3 = {};
    arr1.fill(2);

    zdm::util::fmadd(std::span<int>{arr1}, -1, std::span<const int>{arr3});
    return arr1 == std::array<int, 39>{};
}

bool checkerboard_inner_product_gives_zero()
{
    std::array<int, 39> arr1 = {};
    for (std::size_t i = 0; i < 39; i += 2)
        arr1[i] = 1;
    std::array<int, 39> arr2 = {};
    for (std::size_t i = 1; i < 39; i += 2)
        arr2[i] = 1;
    return zdm::util::inner_product(std::span<const int>{arr1}, std::span<const int>{arr2});
}

} // namespace

int main()
{
    assert(zdm::util::even_floor(0U) == 0);
    assert(zdm::util::even_floor(1U) == 0);
    assert(zdm::util::even_floor(2U) == 2);
    assert(zdm::util::even_floor(3U) == 2);
    assert(zdm::util::even_floor(17U) == 16);
    assert(zdm::util::even_floor(23748U) == 23748);
    assert(zdm::util::even_floor(23749U) == 23748);

    assert(empty_spans_have_no_overlap());
    assert(nonempty_span_overlaps_with_itself());
    assert(spans_of_distinct_arrays_have_no_overlap());

    assert(inplace_product_of_array_with_zero_array_is_zero());
    assert(inplace_product_of_array_with_identity_array_does_not_change_value());
    assert(inplace_product_of_array_with_zero_is_zero());
    assert(inplace_product_of_array_with_identity_does_not_change_value());

    assert(product_of_array_with_zero_array_is_zero());
    assert(product_of_array_with_identity_array_does_not_change_value());
    assert(product_of_array_with_zero_is_zero());
    assert(product_of_array_with_identity_does_not_change_value());

    assert(inplace_fmadd_with_negation_gives_zero());
    assert(inplace_fmadd_with_scalar_negation_gives_zero());

    assert(checkerboard_inner_product_gives_zero());
}
