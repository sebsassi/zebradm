#include "linalg.hpp"

namespace zdm
{

template <typename ParametricTransform, typename OutputType>
concept outputs = requires (ParametricTransform x, typename ParametricTransform::value_type t)
    {
        { x(t) } -> std::same_as<OutputType>;
    };

template <typename T>
concept parametric_rigid_transform
    = outputs<T, RigidTransform<typename T::value_type, T::dimension, T::action, T::matrix_layout>>;

template <typename RigidTransformType, parametric_rigid_transform... Types>
    requires (outputs<Types, RigidTransformType> && ...)
class CompositeRigidTransform
{
public:
    using rigid_transform_type = RigidTransformType;

    CompositeRigidTransform(const Types&... transforms): m_transforms(transforms...) {}

    [[nodiscard]] rigid_transform_type operator()(const rigid_transform_type::value_type& parameter) const noexcept
    {
        // TODO: implementation
    }

private:
    std::tuple<Types...> m_transforms;
};


} // namespace zdm
