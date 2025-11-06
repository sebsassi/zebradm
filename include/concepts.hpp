#pragma once

#include <type_traits>

namespace zdm
{

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

} // namespace zdm
