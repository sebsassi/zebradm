#include "time.hpp"

#include <cassert>

using namespace std::literals;

void test_leap_year()
{
    assert(zdm::time::is_leap_year(0) == true);
    assert(zdm::time::is_leap_year(1) == false);
    assert(zdm::time::is_leap_year(-1) == false);
    assert(zdm::time::is_leap_year(4) == true);
    assert(zdm::time::is_leap_year(-4) == true);
    assert(zdm::time::is_leap_year(100) == false);
    assert(zdm::time::is_leap_year(-100) == false);
    assert(zdm::time::is_leap_year(400) == false);
    assert(zdm::time::is_leap_year(-400) == false);

    assert(zdm::time::is_leap_year(1900) == false);
    assert(zdm::time::is_leap_year(1972) == true);
    assert(zdm::time::is_leap_year(2000) == true);
    assert(zdm::time::is_leap_year(2009) == false);
    assert(zdm::time::is_leap_year(2012) == true);
}

void test_days_in_year()
{
    assert(zdm::time::days_in_year(2000) == 366);
    assert(zdm::time::days_in_year(2001) == 365);
}

void test_day_of_year()
{
    assert(zdm::time::day_of_year(2000, 0) == 0);
    assert(zdm::time::day_of_year(2001, 0) == 0);
    assert(zdm::time::day_of_year(2000, 1) == 31);
    assert(zdm::time::day_of_year(2001, 1) == 31);
    assert(zdm::time::day_of_year(2000, 2) == 59);
    assert(zdm::time::day_of_year(2001, 2) == 60);
    assert(zdm::time::day_of_year(2000, 3) == 90);
    assert(zdm::time::day_of_year(2001, 3) == 91);
    assert(zdm::time::day_of_year(2000, 4) == 120);
    assert(zdm::time::day_of_year(2001, 4) == 121);
    assert(zdm::time::day_of_year(2000, 5) == 151);
    assert(zdm::time::day_of_year(2001, 5) == 152);
    assert(zdm::time::day_of_year(2000, 6) == 181);
    assert(zdm::time::day_of_year(2001, 6) == 182);
    assert(zdm::time::day_of_year(2000, 7) == 212);
    assert(zdm::time::day_of_year(2001, 7) == 213);
    assert(zdm::time::day_of_year(2000, 8) == 243);
    assert(zdm::time::day_of_year(2001, 8) == 244);
    assert(zdm::time::day_of_year(2000, 9) == 273);
    assert(zdm::time::day_of_year(2001, 9) == 274);
    assert(zdm::time::day_of_year(2000, 10) == 304);
    assert(zdm::time::day_of_year(2001, 10) == 305);
    assert(zdm::time::day_of_year(2000, 11) == 334);
    assert(zdm::time::day_of_year(2001, 11) == 335);

    assert(zdm::time::day_of_year(2000, 0, 1) == 1);
}

bool month_of_year_gives_correct_month_and_day_of_month_for(std::int32_t year, std::uint32_t day_of_year, std::uint32_t real_month, std::uint32_t real_day_of_month)
{
    const auto& [month, day_of_month] = zdm::time::month_of_year(year, day_of_year);
    return month == real_month && day_of_month == real_day_of_month;
}

bool month_of_year_is_inverse_of_day_of_year(std::int32_t year, std::uint32_t day_of_year)
{
    const auto& [month, day_of_month] = zdm::time::month_of_year(year, day_of_year);
    return zdm::time::day_of_year(year, month, day_of_month) == day_of_year;
}

void test_month_of_year()
{
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 1, 0, 1));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 1, 0, 1));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 15, 0, 15));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 15, 0, 15));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 45, 1, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 45, 1, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 60, 1, 29));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 60, 2, 1));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 75, 2, 15));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 75, 2, 16));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 105, 3, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 105, 3, 15));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 135, 4, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 135, 4, 15));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 165, 5, 13));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 165, 5, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 195, 6, 13));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 195, 6, 14));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 225, 7, 12));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 225, 7, 13));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 255, 8, 11));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 255, 8, 12));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 285, 9, 11));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 285, 9, 12));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 315, 10, 10));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 315, 10, 11));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 345, 10, 10));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 345, 10, 11));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2000, 366, 11, 31));
    assert(month_of_year_gives_correct_month_and_day_of_month_for(2001, 365, 11, 31));
    
    assert(month_of_year_is_inverse_of_day_of_year(2000, 1));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 1));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 15));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 15));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 45));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 45));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 60));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 60));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 75));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 75));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 105));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 105));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 135));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 135));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 165));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 165));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 195));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 195));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 225));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 225));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 255));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 255));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 285));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 285));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 315));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 315));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 345));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 345));
    assert(month_of_year_is_inverse_of_day_of_year(2000, 366));
    assert(month_of_year_is_inverse_of_day_of_year(2001, 365));
}

void test_days_until()
{
    assert(zdm::time::days_until(0) == 0);
    assert(zdm::time::days_until(1) == 366);
    assert(zdm::time::days_until(2) == 366 + 365);
    assert(zdm::time::days_until(3) == 366 + 2*365);
    assert(zdm::time::days_until(4) == 366 + 3*365);
    assert(zdm::time::days_until(5) == 2*366 + 3*365);
    assert(zdm::time::days_until(100) == 100*365 + 25);
    assert(zdm::time::days_until(101) == 100*365 + 25 + 365);
    assert(zdm::time::days_until(400) == 400*365 + 100 - 3);
    assert(zdm::time::days_until(401) == 400*365 + 100 - 3 + 366);
}

void test_time_comparison_operations()
{
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} == zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));

    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2001, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2000, 2, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2000, 1, 2, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2000, 1, 1, 1, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2000, 1, 1, 0, 1, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 0, 0} < zdm::time::Time{2000, 1, 1, 0, 0, 1, 0}));

    assert((zdm::time::Time{2001, 1, 1, 0, 0, 0, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 2, 1, 0, 0, 0, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 2, 0, 0, 0, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 1, 0, 0, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 1, 0, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
    assert((zdm::time::Time{2000, 1, 1, 0, 0, 1, 0} > zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}));
}

bool test_time_add_time_zone_offset_gives_correct_time(zdm::time::Time time, zdm::time::TimeZoneOffset offset, zdm::time::Time offset_time)
{
    return time.add(offset) == offset_time;
}

void test_time_add_time_zone_offset()
{
    test_time_add_time_zone_offset_gives_correct_time(zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}, zdm::time::TimeZoneOffset{+1, 0, 0}, zdm::time::Time{2000, 1, 1, 0, 0, 0, 0});
    test_time_add_time_zone_offset_gives_correct_time(zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}, zdm::time::TimeZoneOffset{+1, 4, 30}, zdm::time::Time{2000, 1, 1, 4, 30, 0, 0});
    test_time_add_time_zone_offset_gives_correct_time(zdm::time::Time{2000, 1, 1, 0, 0, 0, 0}, zdm::time::TimeZoneOffset{-1, 4, 30}, zdm::time::Time{1999, 12, 31, 19, 30, 0, 0});
    test_time_add_time_zone_offset_gives_correct_time(zdm::time::Time{1999, 12, 31, 19, 30, 0, 0}, zdm::time::TimeZoneOffset{+1, 4, 30}, zdm::time::Time{2000, 1, 1, 0, 0, 0, 0});
}

void test_find_next_non_whitespace()
{
    assert(zdm::time::detail::find_next_non_whitespace(""sv) == ""sv);
    assert(zdm::time::detail::find_next_non_whitespace("      "sv) == ""sv);
    assert(zdm::time::detail::find_next_non_whitespace(" hello"sv) == "hello"sv);
    assert(zdm::time::detail::find_next_non_whitespace(" \f\n\r\t\v"sv) == ""sv);
    assert(zdm::time::detail::find_next_non_whitespace(" \f\n\r\t\vhello"sv) == "hello"sv);
}

bool test_parse_unsigned_gives_correct_result(std::string_view input, std::uint64_t expected_number, std::string_view expected_output)
{
    const auto result = zdm::time::detail::parse_unsigned(input);
    if (!result.has_value()) return false;

    const auto& [number, output] = *result;
    return number == expected_number && output == expected_output;
}

void test_parse_unsigned()
{
    assert(zdm::time::detail::parse_unsigned(""sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_unsigned("hello"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_unsigned("-1"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_unsigned("+1"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(test_parse_unsigned_gives_correct_result("237489"sv, 237489, ""sv));
    assert(test_parse_unsigned_gives_correct_result("237489 hello"sv, 237489, " hello"sv));
}

bool test_parse_signed_gives_correct_result(std::string_view input, std::int64_t expected_number, std::string_view expected_output)
{
    const auto result = zdm::time::detail::parse_signed(input);
    if (!result.has_value()) return false;

    const auto& [number, output] = *result;
    return number == expected_number && output == expected_output;
}

void test_parse_signed()
{
    assert(zdm::time::detail::parse_signed(""sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_signed("hello"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_signed("-hello"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::detail::parse_signed("+hello"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(test_parse_signed_gives_correct_result("237489", 237489, ""sv));
    assert(test_parse_signed_gives_correct_result("-237489", -237489, ""sv));
    assert(test_parse_signed_gives_correct_result("+237489", 237489, ""sv));
    assert(test_parse_signed_gives_correct_result("-237489 a", -237489, " a"sv));
}

bool test_parse_time_zone_offset_gives_correct_result(std::string_view input, zdm::time::TimeZoneOffset expected_offset, std::string_view expected_output)
{
    const auto result = zdm::time::detail::parse_time_zone_offset(input);
    if (!result.has_value()) return false;

    const auto& [offset, output] = *result;
    return offset == expected_offset && output == expected_output;
}

void test_parse_time_zone_offset()
{
    assert(zdm::time::detail::parse_time_zone_offset(""sv) == std::unexpected(zdm::time::DateParseStatus::invalid_time_zone_offset));
    assert(zdm::time::detail::parse_time_zone_offset("hello"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_time_zone_offset));
    assert(zdm::time::detail::parse_time_zone_offset("-hello"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_time_zone_offset));
    assert(zdm::time::detail::parse_time_zone_offset("+hello"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_time_zone_offset));
    assert(zdm::time::detail::parse_time_zone_offset("12:00"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_time_zone_offset));

    assert(test_parse_time_zone_offset_gives_correct_result("Z"sv, zdm::time::TimeZoneOffset{}, ""sv));
    assert(test_parse_time_zone_offset_gives_correct_result("Zhello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
    assert(test_parse_time_zone_offset_gives_correct_result("Zhello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
    assert(test_parse_time_zone_offset_gives_correct_result("+12hello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
    assert(test_parse_time_zone_offset_gives_correct_result("-12hello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
    assert(test_parse_time_zone_offset_gives_correct_result("+1200hello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
    assert(test_parse_time_zone_offset_gives_correct_result("+12:00hello"sv, zdm::time::TimeZoneOffset{}, "hello"sv));
}

bool test_parse_time_is_correct(std::string_view time_string, std::string_view format, zdm::time::Time expected_time, std::string_view expected_output)
{
    const auto result = zdm::time::parse_time(time_string, format);
    if (!result.has_value()) return false;

    const auto& [time, output] = *result;
    return time == expected_time && output == expected_output;
}

void test_parse_time()
{
    assert(zdm::time::parse_time(""sv, "%H:%M:%S"sv) == std::unexpected(zdm::time::DateParseStatus::incomplete_time_string));
    assert(zdm::time::parse_time("hello"sv, "%H:%M:%S"sv) == std::unexpected(zdm::time::DateParseStatus::incomplete_time_string));
    assert(zdm::time::parse_time("12:00"sv, "%H:%M:%S"sv) == std::unexpected(zdm::time::DateParseStatus::incomplete_time_string));
    assert(zdm::time::parse_time("at 12:00"sv, "At %H:%M"sv) == std::unexpected(zdm::time::DateParseStatus::character_mismatch));
    assert(zdm::time::parse_time("12:00"sv, "%H:%"sv) == std::unexpected(zdm::time::DateParseStatus::incomplete_format_string));

    // Check all printable ASCII characters which are not used as format specifiers.
    assert(zdm::time::parse_time("12:00"sv, "%H:% "sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%!"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%\""sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%#"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%$"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%&"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%'"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%("sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%)"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%*"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%+"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%,"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%-"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%."sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%/"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%0"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%1"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%2"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%3"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%4"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%5"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%6"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%7"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%8"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%9"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%:"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%;"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%<"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%="sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%>"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%?"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%@"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%A"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // B used
    assert(zdm::time::parse_time("12:00"sv, "%H:%C"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%D"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%E"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%F"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%G"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // H used
    // I used
    assert(zdm::time::parse_time("12:00"sv, "%H:%J"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%K"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%L"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // M used
    assert(zdm::time::parse_time("12:00"sv, "%H:%N"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%O"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%P"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%Q"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%R"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // S used
    assert(zdm::time::parse_time("12:00"sv, "%H:%T"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%U"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%V"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%W"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%X"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // Y used
    assert(zdm::time::parse_time("12:00"sv, "%H:%Z"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%["sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%\\"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%]"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%^"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%_"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%`"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%a"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // b used
    assert(zdm::time::parse_time("12:00"sv, "%H:%c"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // d used
    // e used
    assert(zdm::time::parse_time("12:00"sv, "%H:%f"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%g"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%h"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%i"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%j"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%k"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%l"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // m used
    // n used
    assert(zdm::time::parse_time("12:00"sv, "%H:%o"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // p used
    assert(zdm::time::parse_time("12:00"sv, "%H:%q"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%r"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%s"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%t"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%u"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%v"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%w"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%x"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%y"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    // z used
    assert(zdm::time::parse_time("12:00"sv, "%H:%{"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%|"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%}"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));
    assert(zdm::time::parse_time("12:00"sv, "%H:%~"sv) == std::unexpected(zdm::time::DateParseStatus::unsupported_format_specifier));

    assert(zdm::time::parse_time("May June"sv, "%b %B"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("may"sv, "%b"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_month_of_year));
    assert(zdm::time::parse_time("Juvember"sv, "%b"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_month_of_year));

    assert(zdm::time::parse_time("20 21"sv, "%d %e"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("0"sv, "%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("32"sv, "%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("first"sv, "%d"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(zdm::time::parse_time("11 23"sv, "%I %H"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("23 11"sv, "%H %I"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("24"sv, "%H"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_hour));
    assert(zdm::time::parse_time("13"sv, "%I"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_hour));
    assert(zdm::time::parse_time("0"sv, "%I"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_hour));
    assert(zdm::time::parse_time("twelve"sv, "%H"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));
    assert(zdm::time::parse_time("twelve"sv, "%I"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(zdm::time::parse_time("5 May"sv, "%m %b"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("0"sv, "%m"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_month_of_year));
    assert(zdm::time::parse_time("13"sv, "%m"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_month_of_year));
    assert(zdm::time::parse_time("May"sv, "%m"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(zdm::time::parse_time("49 50"sv, "%M %M"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("60"sv, "%M"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_minute));
    assert(zdm::time::parse_time("fifteen"sv, "%M"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(zdm::time::parse_time("AM PM"sv, "%p %p"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("am"sv, "%p"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_am_pm));
    assert(zdm::time::parse_time("a.m."sv, "%p"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_am_pm));

    assert(zdm::time::parse_time("49 50"sv, "%S %S"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("61"sv, "%M"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_second));
    assert(zdm::time::parse_time("fifteen"sv, "%M"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_minute));

    assert(zdm::time::parse_time("2000 2001"sv, "%Y %Y"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));
    assert(zdm::time::parse_time("two thousand"sv, "%Y"sv) == std::unexpected(zdm::time::DateParseStatus::expected_digits));

    assert(zdm::time::parse_time("Z +12:00"sv, "%z %z"sv) == std::unexpected(zdm::time::DateParseStatus::duplicate_format_specifiers));

    assert(zdm::time::parse_time("2000-02-30"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("2001-02-29"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("2001-04-31"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("2001-06-31"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("2001-09-31"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));
    assert(zdm::time::parse_time("2001-11-31"sv, "%Y-%m-%d"sv) == std::unexpected(zdm::time::DateParseStatus::invalid_day_of_month));

    assert(test_parse_time_is_correct("%"sv, "%%"sv, zdm::time::Time{0, 0, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("January"sv, "%b"sv, zdm::time::Time{0, 1, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("February"sv, "%b"sv, zdm::time::Time{0, 2, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("March"sv, "%b"sv, zdm::time::Time{0, 3, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("April"sv, "%b"sv, zdm::time::Time{0, 4, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("May"sv, "%b"sv, zdm::time::Time{0, 5, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("June"sv, "%b"sv, zdm::time::Time{0, 6, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("July"sv, "%b"sv, zdm::time::Time{0, 7, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("August"sv, "%b"sv, zdm::time::Time{0, 8, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("September"sv, "%b"sv, zdm::time::Time{0, 9, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("October"sv, "%b"sv, zdm::time::Time{0, 10, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("November"sv, "%b"sv, zdm::time::Time{0, 11, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("December"sv, "%b"sv, zdm::time::Time{0, 12, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("Jan"sv, "%b"sv, zdm::time::Time{0, 1, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Feb"sv, "%b"sv, zdm::time::Time{0, 2, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Mar"sv, "%b"sv, zdm::time::Time{0, 3, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Apr"sv, "%b"sv, zdm::time::Time{0, 4, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("May"sv, "%b"sv, zdm::time::Time{0, 5, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Jun"sv, "%b"sv, zdm::time::Time{0, 6, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Jul"sv, "%b"sv, zdm::time::Time{0, 7, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Aug"sv, "%b"sv, zdm::time::Time{0, 8, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Sep"sv, "%b"sv, zdm::time::Time{0, 9, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Oct"sv, "%b"sv, zdm::time::Time{0, 10, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Nov"sv, "%b"sv, zdm::time::Time{0, 11, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("Dec"sv, "%b"sv, zdm::time::Time{0, 12, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("15"sv, "%d"sv, zdm::time::Time{0, 0, 15, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("22"sv, "%H"sv, zdm::time::Time{0, 0, 0, 22, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("11"sv, "%I"sv, zdm::time::Time{0, 0, 0, 11, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("12"sv, "%I"sv, zdm::time::Time{0, 0, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("10"sv, "%m"sv, zdm::time::Time{0, 10, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("55"sv, "%M"sv, zdm::time::Time{0, 0, 0, 0, 55, 0, 0}, ""sv));

    assert(test_parse_time_is_correct(" \f\n\r\t\v"sv, "%n"sv, zdm::time::Time{0, 0, 0, 0, 0, 55, 0}, ""sv));

    assert(test_parse_time_is_correct("55"sv, "%S"sv, zdm::time::Time{0, 0, 0, 0, 0, 55, 0}, ""sv));

    assert(test_parse_time_is_correct("2000"sv, "%Y"sv, zdm::time::Time{2000, 0, 0, 0, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("09:00 AM"sv, "at %H:%M %p"sv, zdm::time::Time{0, 0, 0, 9, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("12:00 AM"sv, "at %H:%M %p"sv, zdm::time::Time{0, 0, 0, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("09:00 PM"sv, "at %H:%M %p"sv, zdm::time::Time{0, 0, 0, 21, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("12:00 PM"sv, "at %H:%M %p"sv, zdm::time::Time{0, 0, 0, 12, 0, 0, 0}, ""sv));

    assert(test_parse_time_is_correct("12:00:00"sv, "at %H:%M:%S"sv, zdm::time::Time{0, 0, 0, 12, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("June 11, 2005"sv, "%b %e, %Y"sv, zdm::time::Time{2005, 7, 11, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("June 11,   \n2005"sv, "%b %e,%n%Y"sv, zdm::time::Time{2005, 7, 11, 0, 0, 0, 0}, ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45"sv, "%Y-%m-%dT%H:%M:%S"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}, ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45Z"sv, "%Y-%m-%dT%H:%M:%S%z"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}.add(zdm::time::TimeZoneOffset{}), ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45+04"sv, "%Y-%m-%dT%H:%M:%S%z"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}.add(zdm::time::TimeZoneOffset{1, 4, 0}), ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45+0430"sv, "%Y-%m-%dT%H:%M:%S%z"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}.add(zdm::time::TimeZoneOffset{1, 4, 30}), ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45+04:30"sv, "%Y-%m-%dT%H:%M:%S%z"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}.add(zdm::time::TimeZoneOffset{1, 4, 30}), ""sv));
    assert(test_parse_time_is_correct("2025-11-15T09:15:45-04:30"sv, "%Y-%m-%dT%H:%M:%S%z"sv, zdm::time::Time{2025, 11, 15, 9, 15, 45, 0}.add(zdm::time::TimeZoneOffset{-1, 4, 30}), ""sv));

    assert(test_parse_time_is_correct("June 11, 2005 was a nice day"sv, "%b %e, %Y"sv, zdm::time::Time{2005, 7, 11, 0, 0, 0, 0}, " was a nice day"sv));
}

int main()
{
    test_leap_year();
    test_days_in_year();
    test_day_of_year();
    test_month_of_year();
    test_days_until();
    test_time_comparison_operations();
    test_time_add_time_zone_offset();
    test_find_next_non_whitespace();
    test_parse_unsigned();
    test_parse_signed();
    test_parse_time_zone_offset();
    test_parse_time();
}
