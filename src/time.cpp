#include "time.hpp"

namespace zdm
{

[[nodiscard]] std::expected<std::vector<double>, DateParseError>
ut1_j2000_interval(
    std::string_view start_date, std::string_view end_date, std::size_t count, 
    std::string_view format)
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
