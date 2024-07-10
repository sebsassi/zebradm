#pragma once

#include <string>
#include <vector>
#include <ctime>

namespace coordinates
{

class Epoch
{
public:
    constexpr Epoch(const char* p_label, double p_ce_offset) : m_label(p_label), m_ce_offset(p_ce_offset) {}

    [[nodiscard]] constexpr const char* label() const noexcept
    {
        return m_label;
    }

    [[nodiscard]] constexpr double ce_offset() const noexcept
    {
        return m_ce_offset;
    }

    [[nodiscard]] std::vector<double> make_time_interval(
        const std::string& start_date, const std::string& end_date,
        std::size_t size);
private:
    const char* m_label;
    double m_ce_offset;
};

[[nodiscard]] constexpr double days_in_years(double year) noexcept
{
    return 365.0*year + std::ceil(0.25*year) - std::ceil(0.01*year) + std::ceil(0.0025*year);
}

[[nodiscard]] constexpr double tm_to_double(const std::tm& time) noexcept
{
    const double year = double(time.tm_year);
    const double yday = double(time.tm_yday);
    const double hour = double(time.tm_hour);
    const double min = double(time.tm_min);
    const double sec = double(time.tm_sec);
    const double days = days_in_years(year) + yday - 1.0;
    const double fractional_day = (1.0/24.0)*hour + (1.0/1440.0)*min + (1.0/86400.0)*sec;
    return days + fractional_day;
}

constexpr Epoch J2000_epoch()
{
    return Epoch("J2000", 730485.5);
}

}