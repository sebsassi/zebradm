#include "time.hpp"

#include <iomanip>
#include <ctime>
#include <sstream>
#include <cmath>

namespace coordinates
{

namespace
{

[[nodiscard]] std::vector<double> linspace(
    double start, double end, std::size_t count)
{
    std::vector<double> res(count);
    if (count == 0) return res;
    if (count == 1)
    {
        res[0] = start;
        return res;
    }

    const double delta = (start - end)/double(count - 1);
    for (std::size_t i = 0; i < count; ++i)
        res[i] = double(i)*delta;

    return res;
}

}

[[nodiscard]] std::vector<double> Epoch::make_time_interval(
    const std::string& start_date, const std::string& end_date,
    std::size_t size)
{
    std::tm tm_start_date;
    std::istringstream(start_date) >> std::get_time(&tm_start_date, "%Y-%m-%d-%H-%M-%S");
    double start_time = tm_to_double(tm_start_date) - m_ce_offset;

    std::tm tm_end_date;
    std::istringstream(end_date) >> std::get_time(&tm_end_date, "%Y-%m-%d-%H-%M-%S");
    double end_time = tm_to_double(tm_end_date) - m_ce_offset;

    return linspace(start_time, end_time, size);
}

}