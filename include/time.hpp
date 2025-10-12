#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

namespace zdm
{

enum class DateParseError
{
    incomplete_format_string,
    character_mismatch,
    unsupported_format_specifier,
    duplicate_format_specifiers,
    expected_digits,
    invalid_day_of_month,
    invalid_month_of_year,
    invalid_hour,
    invalid_minute,
    invalid_second,
    invalid_am_pm,
    invalid_time_zone_offset
};

namespace detail
{

enum class FormatSpecifier : unsigned char
{
    unsupported = 0,
    percentage,         // %%
    // week_day_name,      // %a, %A
    month_name,         // %b, %B
    // century,            // %C
    day_of_month,       // %e, %d
    hour_24,            // %H
    hour_12,            // %I
    // day_of_year,        // %j
    month_of_year,      // %m
    minute,             // %M
    whitespace,         // %n
    am_pm,              // %p
    second,             // %S
    // week_day_mon_1_7,   // %u
    // week_of_year_sun,   // %U
    // week_day_sun_0_6,   // %w
    // week_of_year_mon,   // %W
    // year_in_century,    // %y
    year,               // %Y
    time_zone_offset,   // %z
};

enum class DateParseStatusFlag : std::uint32_t
{
    has_day_of_month = (1UL << 1UL),
    has_month_of_year = (1UL << 2UL),
    has_year = (1UL << 3UL),
    has_hour = (1UL << 4UL),
    has_minute = (1UL << 5UL),
    has_second = (1UL << 6UL),
    has_am_pm = (1UL << 7UL),
    has_time_zone_offset = (1UL << 8UL),
    is_pm = (1UL << 9UL),
    needs_am_pm = (1UL << 10UL),
};

[[nodiscard]] consteval std::array<FormatSpecifier, 256> format_specifier_map()
{
    std::array<FormatSpecifier, 256> map = {};
    map['%'] = FormatSpecifier::percentage;
    // map['a'] = FormatSpecifier::week_day_name;
    // map['A'] = FormatSpecifier::week_day_name;
    map['b'] = FormatSpecifier::month_name;
    map['B'] = FormatSpecifier::month_name;
    // map['C'] = FormatSpecifier::century;
    map['d'] = FormatSpecifier::day_of_month;
    map['e'] = FormatSpecifier::day_of_month;
    map['H'] = FormatSpecifier::hour_24;
    map['I'] = FormatSpecifier::hour_12;
    // map['j'] = FormatSpecifier::day_of_year;
    map['m'] = FormatSpecifier::month_of_year;
    map['M'] = FormatSpecifier::minute;
    map['n'] = FormatSpecifier::whitespace;
    map['p'] = FormatSpecifier::am_pm;
    map['S'] = FormatSpecifier::second;
    // map['u'] = FormatSpecifier::week_day_mon_1_7;
    // map['U'] = FormatSpecifier::week_of_year_sun;
    // map['w'] = FormatSpecifier::week_day_sun_0_6;
    // map['W'] = FormatSpecifier::week_of_year_mon;
    // map['y'] = FormatSpecifier::year_in_century;
    map['Y'] = FormatSpecifier::year;
    map['z'] = FormatSpecifier::time_zone_offset;
    return map;
}

[[nodiscard]] consteval std::array<std::array<std::uint32_t, 12>, 2>
days_in_year_before_month()
{
    return {
        std::array<std::uint32_t, 12>{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
        std::array<std::uint32_t, 12>{0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}
    };
}

} // namespace detail

/**
    @brief Check if year is a leap year

    @param year

    @return true if the year is a leap year.
*/
template <std::integral T>
[[nodiscard]] constexpr bool
is_leap_year(T year) noexcept
{
    return (year % 4) == 0 && (year % 100 != 0 || year % 400 == 0);
}

/**
    @brief Number of days in a given year

    @param year

    @return number of days in the year.
*/
[[nodiscard]] constexpr std::uint32_t
days_in_year(std::int32_t year) noexcept
{
    return 365 + std::uint32_t(is_leap_year(year));
}

/**
    @brief Computes the day of year corresponding to the first day of a given
    month.

    @param year
    @param month

    @return day of year (1-366).
*/
[[nodiscard]] constexpr std::uint32_t
day_of_year(std::int32_t year, std::uint32_t month) noexcept
{
    constexpr std::array<std::array<std::uint32_t, 12>, 2>
    days_before_month = detail::days_in_year_before_month();
    
    return days_before_month[std::size_t(is_leap_year(year))][month];
}

/**
    @brief Computes the day of year corresponding to a given month and day of
    month. Inverse of `month_of_year`.

    @param year
    @param month
    @param day_of_month

    @return day of year (1-366).
*/
[[nodiscard]] constexpr std::uint32_t
day_of_year(std::int32_t year, std::uint32_t month, std::uint32_t day_of_month) noexcept
{
    return day_of_year(year, month) + day_of_month;
}

/**
    @brief Computes the month and day of month of the given year, given a day
    of year. Inverse of `day_of_year`.

    @param year
    @param day_of_year

    @return month number (0-11) and day of month (1-31).
*/
[[nodiscard]] constexpr std::pair<std::uint32_t, std::uint32_t>
month_of_year(std::int32_t year, std::uint32_t day_of_year)
{
    constexpr std::array<std::array<std::uint32_t, 12>, 2>
    days_before_month = detail::days_in_year_before_month();

    std::uint32_t month = 0;
    const auto leap_year = std::uint32_t(is_leap_year(year));
    for (; month < 12; ++month)
    {
        if (days_before_month[leap_year][month] < day_of_year) break;
    }
    return {month, day_of_year - days_before_month[leap_year][month]};
}

/**
    @brief Counts the number of days from the start of year 0 until start of
    the given year.

    @param year

    @return number of days since start of year 0 until start of year.
*/
[[nodiscard]] constexpr std::int64_t
days_until(std::int32_t year) noexcept
{
    const auto y64 = std::int64_t(year);
    return y64*365 + y64/4 - y64/100 + y64/400;
}

/**
    @brief Structure representing a time zone offset.
*/
struct TimeZoneOffset
{
    std::int32_t sign = 1;
    std::uint32_t hour;
    std::uint32_t min;
};

/**
    @brief Structure representing a point in time.

    This structure represents a point in time specified by a year, month, day
    of month, hour, minute, and second. This structure closely resembles the `tm`
    structure found in the C standard library, but differs from it in some
    notable ways. Most notably, the year in this structure is the actual year,
    and not a year relative to 1900. Furthermore, the data layout is different,
    and this structure doesn't have `wday`, `yday`, and `isdst` fields.
*/
struct Time
{
    std::int32_t year;
    std::uint16_t mon;
    std::uint16_t mday;
    std::uint16_t hour;
    std::uint16_t min;
    std::uint16_t sec;


    /**
        @brief Apply a time zone offset to a time.

        @param offset

        @return time with the time zone offset applied.
    */
    [[nodiscard]] constexpr Time
    convert(TimeZoneOffset offset) const noexcept
    {
        const std::uint32_t month = mon;
        const std::uint32_t day_of_month = mday;
        const auto second_of_year = std::int32_t((day_of_year(year, month, day_of_month) - 1)*86400LL + hour*3600LL + min*60LL + sec);
        const std::int32_t offset_seconds = offset.sign*60*std::int32_t(60*offset.hour + offset.min);

        std::int32_t second_of_offset_year = second_of_year + offset_seconds;
        auto seconds_in_current_year = std::int32_t(6400*days_in_year(year));
        std::int32_t offset_year = year;
        if (second_of_offset_year > seconds_in_current_year)
        {
            ++offset_year;
            second_of_offset_year -= seconds_in_current_year;
        }
        if (second_of_offset_year < 0)
        {
            const auto seconds_in_last_year = std::int32_t(86400*days_in_year(year - 1));
            --offset_year;
            second_of_offset_year += seconds_in_last_year;
        }

        std::uint32_t minute_of_offset_year = std::uint32_t(second_of_offset_year)/60;
        std::uint32_t hour_of_offset_year = minute_of_offset_year/60;
        std::uint32_t day_of_offset_year = hour_of_offset_year/24;
        const auto& [month_of_offset_year, day_of_month_offset] = month_of_year(year, day_of_offset_year);

        return {
            offset_year,
            std::uint16_t(month_of_offset_year),
            std::uint16_t(day_of_month_offset),
            std::uint16_t(hour_of_offset_year % 60),
            std::uint16_t(minute_of_offset_year % 60),
            std::uint16_t(second_of_offset_year % 60)
        };
    }

    /**
        @brief Convert a time to seconds since the zero point of the time struct.
    */
    [[nodiscard]] constexpr std::int64_t to_seconds() const noexcept
    {
        std::uint32_t month = mon;
        std::uint32_t day_of_month = mday;
        return days_until(year)*86400LL + std::int64_t(day_of_year(year, month, day_of_month) - 1)*86400LL + hour*3600LL + min*60LL + sec;
    }
};

/**
    @brief Compute the number of seconds that have passed since a given epoch

    @tparam epoch

    @param time

    @return number of seconds since the epoch.
*/
template <Time epoch>
[[nodiscard]] constexpr std::int64_t seconds_since_epoch(Time time) noexcept
{
    constexpr std::int64_t epoch_seconds = epoch.to_seconds();
    return time.to_seconds() - epoch_seconds;
}

namespace detail
{

[[nodiscard]] constexpr std::expected<std::pair<std::uint64_t, std::size_t>, DateParseError>
parse_unsigned(std::string_view str) noexcept
{
    if (str[0] <= '0' || '9' <= str[0])
        return std::unexpected(DateParseError::expected_digits);

    auto value = std::uint64_t(str[0] - '0');
    std::size_t length = 1;
    for (; '0' <= str[length] && str[length] <= '9'; ++length)
    {
        value *= 10;
        value += uint64_t(str[length] - '0');
    }

    return std::pair{value, length};
}

[[nodiscard]] constexpr std::expected<std::pair<std::int64_t, std::size_t>, DateParseError>
parse_signed(std::string_view str) noexcept
{
    std::int64_t sign = 1;
    std::size_t sign_offset = 0;
    if (str[0] == '-')
    {
        sign = -1;
        str = str.substr(1);
        sign_offset = 1;
    }
    if (str[0] == '+')
    {
        str = str.substr(1);
        sign_offset = 1;
    }

    return parse_unsigned(str)
        .transform([&](auto x) {
            return std::pair{sign*std::int64_t(x.first), x.second + sign_offset};
        });
}

[[nodiscard]] constexpr std::expected<std::pair<TimeZoneOffset, std::size_t>, DateParseError>
parse_time_zone_offset(std::string_view str) noexcept
{
    if (str[0] == 'Z')
        return std::pair{TimeZoneOffset{}, 1};

    std::int32_t sign = 1;
    std::size_t length = 0;
    if (str[0] == '-')
    {
        sign = -1;
        str = str.substr(1);
        length = 1;
    }
    if (str[0] == '+')
    {
        str = str.substr(1);
        length = 1;
    }

    if (str.size() < 2)
        return std::unexpected(DateParseError::invalid_time_zone_offset);
    if (str[0] < '0' || '9' < str[0] || str[1] < '0' || '9' < str[1])
        return std::unexpected(DateParseError::invalid_time_zone_offset);
    length += 2;
    std::uint32_t hour = std::uint32_t(str[0] - '0')*10 + std::uint32_t(str[1] - '0');
    // NOTE:I'm not going to check the hour. If you want to found a country
    // with +99:59 offset, the ISO may not like you, but I won't discriminate.

    if (str[2] != ':' && (str[2]< '0' || '9' < str[2]))
        return std::pair{TimeZoneOffset{sign, hour, 0}, length};
    if (str[2] == ':')
    {
        str = str.substr(3);
        ++length;
    }

    if (str.size() < 2)
        return std::unexpected(DateParseError::invalid_time_zone_offset);
    if (str[0] < '0' || '9' < str[0] || str[1] < '0' || '9' < str[1])
        return std::unexpected(DateParseError::invalid_time_zone_offset);
    length += 2;
    std::uint32_t min = std::uint32_t(str[0] - '0')*10 + std::uint32_t(str[1] - '0');
    if (min > 59)
        return std::unexpected(DateParseError::invalid_time_zone_offset);

    return std::pair{TimeZoneOffset{sign, hour, min}, length};
}

struct DateParseState
{
    std::uint32_t flags;
    TimeZoneOffset time_zone_offset;
};

} // namespace detail

/**
    @brief Parse a time string to a struct.

    @param input time string to parse.
    @param format string specifying the format of the time string.

    @return time as a struct specifying a date, or a `DateParseError`.

    This function parses a date string to a `Time` struct according to
    the format string. The format string supports the following subset of
    conventional format specifiers:
        - %%: Percentage sign.
        - %b or %B: English name of month, full or abbreviated.
        - %d or %e: Day of month (1-31).
        - %H: Hour on 24-hour clock (0-23).
        - %I: Hour on 12-hour clock (1-12).
        - %m: Month number (1-12).
        - %n: Any whitespace.
        - %p: "AM" or "PM".
        - %S: Second (0-60).
        - %Y: Year.
        - %z: ISO 8601 time zone offset.
*/
constexpr std::expected<std::pair<Time, std::string_view>, DateParseError>
parse_time(std::string_view input, std::string_view format)
{
    constexpr std::array<detail::FormatSpecifier, 256> format_map = detail::format_specifier_map();

    constexpr std::array<std::string_view, 12> month_names
        = {
            "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December"
        };
    constexpr std::array<std::string_view, 12> month_names_abbrv
        = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

    detail::DateParseState parse_state = {};
    Time time = {};

    std::size_t inp_ind = 0;
    for (std::size_t fmt_ind = 0; fmt_ind < format.size(); ++fmt_ind)
    {
        if (std::isspace(format[fmt_ind]))
        {
            while(std::isspace(input[inp_ind])) ++inp_ind;
            continue;
        }
        if (format[fmt_ind] != '%')
        {
            if (format[fmt_ind] != input[inp_ind])
                return std::unexpected(DateParseError::character_mismatch);
            continue;
        }
        if (fmt_ind + 1 >= format.size())
            return std::unexpected(DateParseError::incomplete_format_string);
        ++fmt_ind;

        detail::FormatSpecifier format_specifier = format_map[(unsigned char)(format[fmt_ind])];

        switch (format_specifier)
        {
            case detail::FormatSpecifier::unsupported:
                return std::unexpected(DateParseError::unsupported_format_specifier);

            case detail::FormatSpecifier::percentage:
                continue;

            case detail::FormatSpecifier::month_name:
            {
                std::string_view input_current = input.substr(inp_ind);
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_month_of_year))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                for (std::size_t i = 0; i < month_names.size(); ++i)
                {
                    if (input_current.starts_with(month_names[i]))
                    {
                        parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_month_of_year);
                        time.mon = std::uint16_t(i);
                        inp_ind += month_names[i].size();
                        break;
                    }
                }
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_month_of_year))
                    continue;

                for (std::size_t i = 0; i < month_names_abbrv.size(); ++i)
                {
                    if (input_current.starts_with(month_names_abbrv[i]))
                    {
                        parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_month_of_year);
                        time.mon = std::uint16_t(i);
                        inp_ind += month_names_abbrv[i].size();
                        break;
                    }
                }
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_month_of_year))
                    continue;
                else
                    return std::unexpected(DateParseError::invalid_month_of_year);
            }

            case detail::FormatSpecifier::day_of_month:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_day_of_month))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number < 1 || 31 < number)
                        return std::unexpected(DateParseError::invalid_day_of_month);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_day_of_month);
                    time.mday = std::uint16_t(number);
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::hour_24:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_hour))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number > 23)
                        return std::unexpected(DateParseError::invalid_hour);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_hour);
                    time.hour = std::uint16_t(number);
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::hour_12:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_hour))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number < 1 || 12 < number)
                        return std::unexpected(DateParseError::invalid_hour);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_hour);
                    time.hour = std::uint16_t(number);
                    if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::is_pm))
                    {
                        time.hour += 12;
                        if (time.hour == 24)
                            time.hour = 12;
                    }
                    else if (time.hour == 12)
                        time.hour = 0;

                    if (!(parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_am_pm)))
                        parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::needs_am_pm);
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::month_of_year:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_month_of_year))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number < 1 || 12 < number)
                        return std::unexpected(DateParseError::invalid_month_of_year);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_month_of_year);
                    time.mon = std::uint16_t(number) - 1;
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::minute:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_minute))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number > 60)
                        return std::unexpected(DateParseError::invalid_minute);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_minute);
                    time.min = std::uint16_t(number) - 1;
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::whitespace:
            {
                while (std::isspace(input[inp_ind])) ++inp_ind;
                continue;
            }

            case detail::FormatSpecifier::am_pm:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_am_pm))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                std::string_view input_current = input.substr(inp_ind);
                if (input_current.starts_with("AM"))
                {
                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_am_pm);
                    if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::needs_am_pm))
                    {
                        parse_state.flags ^= std::uint64_t(detail::DateParseStatusFlag::needs_am_pm);
                    }
                    continue;
                }
                if (input_current.starts_with("PM"))
                {
                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_am_pm);
                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::is_pm);
                    if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::needs_am_pm))
                    {
                        parse_state.flags ^= std::uint64_t(detail::DateParseStatusFlag::needs_am_pm);
                        time.hour += 12;
                        if (time.hour == 24)
                            time.hour = 0;
                    }
                    continue;
                }
                return std::unexpected(DateParseError::invalid_am_pm);
            }

            case detail::FormatSpecifier::second:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_second))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_unsigned(input.substr(inp_ind)); results.has_value())
                {
                    const std::uint64_t number = results->first;
                    if (number > 59)
                        return std::unexpected(DateParseError::invalid_minute);

                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_minute);
                    time.sec = std::uint16_t(number) - 1;
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::year:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_year))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);

                if (const auto results = detail::parse_signed(input.substr(inp_ind)); results.has_value())
                {
                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_year);
                    time.year = std::int32_t(results->first);
                    inp_ind += results->second;
                }
                continue;
            }

            case detail::FormatSpecifier::time_zone_offset:
            {
                if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_time_zone_offset))
                    return std::unexpected(DateParseError::duplicate_format_specifiers);


                if (const auto results = detail::parse_time_zone_offset(input.substr(inp_ind)); results.has_value())
                {
                    parse_state.flags |= std::uint64_t(detail::DateParseStatusFlag::has_time_zone_offset);
                    parse_state.time_zone_offset = results->first;
                    inp_ind += results->second;
                }
                continue;
            }
        }
    }

    if (parse_state.flags & std::uint64_t(detail::DateParseStatusFlag::has_time_zone_offset))
        return std::pair{time.convert(parse_state.time_zone_offset), input.substr(inp_ind)};
    else
        return std::pair{time, input.substr(inp_ind)};
}

/**
    @brief Convert a date string to a UT1 time (in days) since the J2000 epoch.

    @param date
    @param format string specifying the format of the date string.

    @return UT1 time since the J2000 epoch in days, or a `DateParseError`.

    @note The format string supports only a subset of common time format
    specifiers. See documentation of the function `parse_time` for list of
    supported format specifiers.
*/
[[nodiscard]] constexpr std::expected<double, DateParseError>
ut1_j2000_from_date(std::string_view date, std::string_view format)
{
    constexpr Time epoch_j2000 = {2000, 0, 1, 12, 0, 0};
    return parse_time(date, format)
        .transform([&](auto val){ return (1.0/86400)*double(seconds_since_epoch<epoch_j2000>(val.first)); });
}

/**
    @brief Generate an interval of UT1 times (in days) since J2000 epoch from date
    strings and a count.

    @param start_date first date of the interval.
    @param end_date last date of the interval.
    @param count number of time points in the interval.
    @param format string specifying the format of the date strings.

    @return vector of UT1 time points in days relative to the J2000 epoch, or a
    `DateParseError`.

    @note The format string supports only a subset of common time format
    specifiers. See documentation of the function `parse_time` for list of
    supported format specifiers.
*/
[[nodiscard]] std::expected<std::vector<double>, DateParseError>
ut1_j2000_interval(std::string_view start_date, std::string_view end_date, std::size_t count, std::string_view format)
{
    if (count == 0) return {};

    const auto start_res = ut1_j2000_from_date(start_date, format);
    if (!start_res.has_value())
        return std::unexpected(start_res.error());

    const double start_time = *start_res;
    
    if (count == 1) return std::vector{start_time};

    const auto end_res = ut1_j2000_from_date(end_date, format);
    if (!end_res.has_value())
        return std::unexpected(end_res.error());

    const double end_time = *end_res;

    const double dt = (start_time - end_time)/double(count - 1);

    std::vector<double> result(count);
    for (std::size_t i = 0; i < count - 1; ++i)
        result[i] = start_time + dt*double(i);
    result[count - 1] = end_time;

    return result;
}

} // namespace zdm
