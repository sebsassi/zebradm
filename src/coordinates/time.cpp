/*
Copyright (c) 2024 Sebastian Sassi

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