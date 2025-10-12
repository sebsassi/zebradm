#include "time.hpp"

#include <cassert>

void test_leap_year()
{
    assert(zdm::is_leap_year(0) == true);
    assert(zdm::is_leap_year(1) == false);
    assert(zdm::is_leap_year(-1) == false);
    assert(zdm::is_leap_year(4) == true);
    assert(zdm::is_leap_year(-4) == true);
    assert(zdm::is_leap_year(100) == false);
    assert(zdm::is_leap_year(-100) == false);
    assert(zdm::is_leap_year(400) == false);
    assert(zdm::is_leap_year(-400) == false);

    assert(zdm::is_leap_year(1900) == false);
    assert(zdm::is_leap_year(1972) == true);
    assert(zdm::is_leap_year(2000) == true);
    assert(zdm::is_leap_year(2009) == false);
    assert(zdm::is_leap_year(2012) == true);
}

void test_days_in_year()
{
    assert(zdm::days_in_year(2000) == 366);
    assert(zdm::days_in_year(2001) == 365);
}

void test_day_of_year()
{
    assert(zdm::day_of_year(2000, 0) == 0);
    assert(zdm::day_of_year(2001, 0) == 0);
    assert(zdm::day_of_year(2000, 1) == 31);
    assert(zdm::day_of_year(2001, 1) == 31);
    assert(zdm::day_of_year(2000, 2) == 59);
    assert(zdm::day_of_year(2001, 2) == 60);
    assert(zdm::day_of_year(2000, 3) == 90);
    assert(zdm::day_of_year(2001, 3) == 91);
    assert(zdm::day_of_year(2000, 4) == 120);
    assert(zdm::day_of_year(2001, 4) == 121);
    assert(zdm::day_of_year(2000, 5) == 151);
    assert(zdm::day_of_year(2001, 5) == 152);
    assert(zdm::day_of_year(2000, 6) == 181);
    assert(zdm::day_of_year(2001, 6) == 182);
    assert(zdm::day_of_year(2000, 7) == 212);
    assert(zdm::day_of_year(2001, 7) == 213);
    assert(zdm::day_of_year(2000, 8) == 243);
    assert(zdm::day_of_year(2001, 8) == 244);
    assert(zdm::day_of_year(2000, 9) == 273);
    assert(zdm::day_of_year(2001, 9) == 274);
    assert(zdm::day_of_year(2000, 10) == 304);
    assert(zdm::day_of_year(2001, 10) == 305);
    assert(zdm::day_of_year(2000, 11) == 334);
    assert(zdm::day_of_year(2001, 11) == 335);
}

int main()
{
    test_leap_year();
    test_days_in_year();
    test_day_of_year();
}
