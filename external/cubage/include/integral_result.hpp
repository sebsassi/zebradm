#pragma once

#include <concepts>
#include <utility>

namespace cubage
{

template <typename FieldType>
concept VectorValued = requires (FieldType& a, FieldType b)
{
    a += b;
    a -= b;
    a + b;
    a - b;
}
&& requires (FieldType a, typename FieldType::value_type c)
{
    a *= c;
    a*c;
    c*a;
};

template <typename FieldType>
concept FloatingPointVectorOperable
    = VectorValued<FieldType> && std::floating_point<typename FieldType::value_type>;

template <typename FieldType>
concept ArrayLike = requires (FieldType x, std::size_t i)
{
    std::tuple_size<FieldType>::value;
    x[i];
};

template <typename ValueType, typename StatusType>
struct Result
{
    using value_type = ValueType;
    using status_type = StatusType;

    value_type value;
    status_type status;
};

enum class Status
{
    SUCCESS,
    MAX_SUBDIV
};

template <typename FieldType>
    requires std::floating_point<FieldType>
        || (ArrayLike<FieldType> && FloatingPointVectorOperable<FieldType>)
struct IntegralResult
{
    FieldType val;
    FieldType err;

    [[nodiscard]] constexpr std::size_t ndim() const noexcept
    {
        if constexpr (std::is_floating_point<FieldType>::value)
            return 1;
        else
            return std::tuple_size<FieldType>::value;
    }

    constexpr IntegralResult& operator+=(const IntegralResult& x) noexcept
    {
        val += x.val;
        err += x.err;
        return *this;
    }

    constexpr IntegralResult& operator-=(const IntegralResult& x) noexcept
    {
        val -= x.val;
        err -= x.err;
        return *this;
    }

    [[nodiscard]] constexpr IntegralResult
    operator+(const IntegralResult& x) const noexcept
    {
        return IntegralResult{this->val + x.val, this->err + x.err};
    }

    [[nodiscard]] constexpr IntegralResult
    operator-(const IntegralResult& x) const noexcept
    {
        return IntegralResult{this->val - x.val, this->err - x.err};
    }
};

}