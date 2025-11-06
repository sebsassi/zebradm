#include "linalg.hpp"
#include "matrix.hpp"
#include "vector.hpp"

#include <cassert>
#include <numbers>

bool is_close(double a, double b, double error)
{
    return std::abs(a - b) < error;
}

template <std::size_t N>
bool is_close(const std::array<double, N>& a, const std::array<double, N>& b, double error)
{
    bool res = true;
    for (std::size_t i = 0; i < N; ++i)
        res = res && is_close(a[i], b[i], error);
    return res;
}

template <std::size_t N>
bool is_close(const zdm::la::Vector<double, N>& a, const zdm::la::Vector<double, N>& b, double error)
{
    return is_close(std::array<double, N>(a), std::array<double, N>(b), error);
}

template <zdm::la::static_square_matrix_like T>
bool is_nearly_orthogonal(T matrix, double error)
{
    using array_type = std::array<typename T::value_type, T::shape[0]*T::shape[1]>;
    const auto maybe_id = zdm::la::matmul(matrix, zdm::la::transpose(matrix));
    const auto id = zdm::la::Matrix<typename T::value_type, T::shape[0], T::shape[1]>::identity();
    return is_close(array_type(maybe_id), array_type(id), error);
}

bool test_matrix_row_major_layout_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major>
    matrix{{1, 2, 3, 4, 5, 6}};

    return matrix[0, 0] == 1 && matrix[0, 1] == 2 && matrix[0, 2] == 3
        && matrix[1, 0] == 4 && matrix[1, 1] == 5 && matrix[1, 2] == 6;
}

bool test_matrix_column_major_layout_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major>
    matrix{{1, 2, 3, 4, 5, 6}};

    return matrix[0, 0] == 1 && matrix[0, 1] == 3 && matrix[0, 2] == 5
        && matrix[1, 0] == 2 && matrix[1, 1] == 4 && matrix[1, 2] == 6;
}

bool test_vector_dot_product_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    return zdm::la::dot(v1, v2) == 1*5 + 2*7 + 3*11;
}

bool test_vector_vector_add_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    return zdm::la::add(v1, v2) == zdm::la::Vector<int, 3>{6, 9, 14};
}

bool test_vector_vector_sub_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    return zdm::la::sub(v1, v2) == zdm::la::Vector<int, 3>{-4, -5, -8};
}

bool test_vector_vector_mul_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    return zdm::la::mul(v1, v2) == zdm::la::Vector<int, 3>{5, 14, 33};
}

bool test_vector_vector_div_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {5, 14, 33};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    return zdm::la::div(v1, v2) == zdm::la::Vector<int, 3>{1, 2, 3};
}

bool test_vector_vector_add_assign_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    zdm::la::add_assign(v1, v2);
    return v1 == zdm::la::Vector<int, 3>{6, 9, 14};
}

bool test_vector_vector_sub_assign_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    zdm::la::sub_assign(v1, v2);
    return v1 == zdm::la::Vector<int, 3>{-4, -5, -8};
}

bool test_vector_vector_mul_assign_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {1, 2, 3};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    zdm::la::mul_assign(v1, v2);
    return v1 == zdm::la::Vector<int, 3>{5, 14, 33};
}

bool test_vector_vector_div_assign_is_correct()
{
    zdm::la::Vector<int, 3> v1 = {5, 14, 33};
    zdm::la::Vector<int, 3> v2 = {5, 7, 11};
    zdm::la::div_assign(v1, v2);
    return v1 == zdm::la::Vector<int, 3>{1, 2, 3};
}

bool test_vector_scalar_add_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::la::add(v, a) == zdm::la::Vector<int, 3>{3, 4, 5};
}

bool test_vector_scalar_sub_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::la::sub(v, a) == zdm::la::Vector<int, 3>{-1, 0, 1};
}

bool test_vector_scalar_mul_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::la::mul(v, a) == zdm::la::Vector<int, 3>{2, 4, 6};
}

bool test_vector_scalar_div_is_correct()
{
    zdm::la::Vector<int, 3> v = {2, 4, 6};
    int a = 2;
    return zdm::la::div(v, a) == zdm::la::Vector<int, 3>{1, 2, 3};
}

bool test_vector_scalar_add_assign_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::la::add_assign(v, a);
    return v == zdm::la::Vector<int, 3>{3, 4, 5};
}

bool test_vector_scalar_sub_assign_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::la::sub_assign(v, a);
    return v == zdm::la::Vector<int, 3>{-1, 0, 1};
}

bool test_vector_scalar_mul_assign_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::la::mul_assign(v, a);
    return v == zdm::la::Vector<int, 3>{2, 4, 6};
}

bool test_vector_scalar_div_assign_is_correct()
{
    zdm::la::Vector<int, 3> v = {2, 4, 6};
    int a = 2;
    zdm::la::div_assign(v, a);
    return v == zdm::la::Vector<int, 3>{1, 2, 3};
}

bool test_scalar_vector_add_is_correct()
{
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::la::add(a, v) == zdm::la::Vector<int, 3>{3, 4, 5};
}

bool test_scalar_vector_sub_is_correct()
{
    int a = 2;
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    return zdm::la::sub(a, v) == zdm::la::Vector<int, 3>{1, 0, -1};
}

bool test_scalar_vector_mul_is_correct()
{
    int a = 2;
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    return zdm::la::mul(a, v) == zdm::la::Vector<int, 3>{2, 4, 6};
}

bool test_scalar_vector_div_is_correct()
{
    int a = 6;
    zdm::la::Vector<int, 3> v = {6, 3, 2};
    return zdm::la::div(a, v) == zdm::la::Vector<int, 3>{1, 2, 3};
}

bool test_matrix_vector_matmul_column_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major> m = {{1, 4, 2, 5, 3, 6}};
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    return zdm::la::matmul(m, v) == zdm::la::Vector<int, 2>{14, 32};
}

bool test_matrix_matrix_matmul_column_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major> m1 = {{1, 4, 2, 5, 3, 6}};
    zdm::la::Matrix<int, 3, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major> m2 = {{1, 3, 5, 2, 4, 6}};
    return zdm::la::matmul(m1, m2) == zdm::la::Matrix<int, 2, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major>{22, 49, 28, 64};
}

bool test_matrix_vector_matmul_row_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major> m = {{1, 2, 3, 4, 5, 6}};
    zdm::la::Vector<int, 3> v = {1, 2, 3};
    return zdm::la::matmul(m, v) == zdm::la::Vector<int, 2>{14, 32};
}

bool test_matrix_matrix_matmul_row_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major> m1 = {{1, 2, 3, 4, 5, 6}};
    zdm::la::Matrix<int, 3, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major> m2 = {{1, 2, 3, 4, 5, 6}};
    return zdm::la::matmul(m1, m2) == zdm::la::Matrix<int, 2, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major>{22, 28, 49, 64};
}

bool test_matrix_transpose_column_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major> m = {{1, 4, 2, 5, 3, 6}};
    return zdm::la::transpose(m) == zdm::la::Matrix<int, 3, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::column_major>{{1, 2, 3, 4, 5, 6}};
}

bool test_matrix_transpose_row_major_is_correct()
{
    zdm::la::Matrix<int, 2, 3, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major> m = {{1, 2, 3, 4, 5, 6}};
    return zdm::la::transpose(m) == zdm::la::Matrix<int, 3, 2, zdm::la::Action::passive, zdm::la::MatrixLayout::row_major>{{1, 4, 2, 5, 3, 6}};
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_2x2_from_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 2, action, layout>::from_angle(0.0);
    return r == zdm::la::RotationMatrix<double, 2, action, layout>::identity();
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_2x2_from_angle_90_nearly_maps_pxpy_to_mypx(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 2, zdm::la::Action::passive, layout>::from_angle(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 2> v = {1.0, 2.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 2>{2.0, -1.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_2x2_from_angle_90_nearly_maps_pxpy_to_pymx(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 2, zdm::la::Action::active, layout>::from_angle(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 2> v = {1.0, 2.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 2>{-2.0, 1.0}, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_x_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_y_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_z_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_x_keeps_x_constant()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const zdm::la::Vector<double, 3> v = {1.0, 0.0, 0.0};
    return zdm::la::matmul(r, v) == v;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_y_keeps_y_constant()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const zdm::la::Vector<double, 3> v = {0.0, 1.0, 0.0};
    return zdm::la::matmul(r, v) == v;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_z_keeps_z_constant()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const zdm::la::Vector<double, 3> v = {0.0, 0.0, 1.0};
    return zdm::la::matmul(r, v) == v;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_coordinate_axis_x_angle_90_nearly_maps_pxpypz_to_pxmzpy(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{1.0, 3.0, -2.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_coordinate_axis_y_angle_90_nearly_maps_pxpypz_to_pzpymx(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template coordinate_axis<zdm::Axis::y>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{-3.0, 2.0, 1.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_coordinate_axis_z_angle_90_nearly_maps_pxpypz_to_mypxpz(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template coordinate_axis<zdm::Axis::z>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{2.0, -1.0, 3.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_coordinate_axis_x_angle_90_nearly_maps_pxpypz_to_pxpzmy(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template coordinate_axis<zdm::Axis::x>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{1.0, -3.0, 2.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_coordinate_axis_y_angle_90_nearly_maps_pxpypz_to_mzpypx(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template coordinate_axis<zdm::Axis::y>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{3.0, 2.0, -1.0}, error);
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_coordinate_axis_z_angle_90_nearly_maps_pxpypz_to_pymxpz(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template coordinate_axis<zdm::Axis::z>(0.5*std::numbers::pi);
    const zdm::la::Vector<double, 3> v = {1.0, 2.0, 3.0};
    return is_close(zdm::la::matmul(r, v), zdm::la::Vector<double, 3>{-2.0, 1.0, 3.0}, error);
}

bool test_passive_rotation_matrix_3x3_coordinate_axis_x_is_nearly_orthogonal(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_nearly_orthogonal(r, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_x_is_nearly_orthogonal(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_nearly_orthogonal(r, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_y_is_nearly_orthogonal(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_nearly_orthogonal(r, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_coordinate_axis_z_is_nearly_orthogonal(double error)
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_nearly_orthogonal(r, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_axis_x_nearly_equals_coordinate_axis_x(double error)
{
    const auto axis = zdm::la::Vector<double, 3>{1.0, 0.0, 0.0};
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::axis(axis, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_axis_y_nearly_equals_coordinate_axis_y(double error)
{
    const auto axis = zdm::la::Vector<double, 3>{0.0, 1.0, 0.0};
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::axis(axis, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_axis_z_nearly_equals_coordinate_axis_z(double error)
{
    const auto axis = zdm::la::Vector<double, 3>{0.0, 0.0, 1.0};
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::axis(axis, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_axis_is_nearly_orthogonal(double error)
{
    const auto axis = zdm::la::Vector<double, 3>{0.5, -0.7, 0.2};
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::axis(axis, 1.3);
    return is_nearly_orthogonal(r, error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xy_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xz_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yx_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yz_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zx_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zy_angle_0_is_identity()
{
    const auto r = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(0.0, 0.0);
    return r == zdm::la::RotationMatrix<double, 3, action, layout>::identity();
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xy_alpha_0_equals_axis_y()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xy_beta_0_equals_axis_x()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xz_alpha_0_equals_axis_z()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xz_beta_0_equals_axis_x()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yx_alpha_0_equals_axis_x()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yx_beta_0_equals_axis_y()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yz_alpha_0_equals_axis_z()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yz_beta_0_equals_axis_y()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zx_alpha_0_equals_axis_x()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zx_beta_0_equals_axis_z()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zy_alpha_0_equals_axis_y()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zy_beta_0_equals_axis_z()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return r1 == r2;
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xy_nearly_equals_axis_x_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xz_nearly_equals_axis_x_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yx_nearly_equals_axis_y_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yz_nearly_equals_axis_y_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zx_nearly_equals_axis_z_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zy_nearly_equals_axis_z_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r23 = zdm::la::matmul(r2, r3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r23), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyx_beta_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyx_alpha_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyx_alpha_beta_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzx_beta_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzx_alpha_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzx_alpha_beta_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxy_beta_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxy_alpha_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxy_alpha_beta_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzy_beta_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzy_alpha_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzy_alpha_beta_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxz_beta_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxz_alpha_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxz_alpha_beta_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyz_beta_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyz_alpha_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyz_alpha_beta_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyz_beta_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyz_alpha_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyz_alpha_beta_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzy_beta_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzy_alpha_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzy_alpha_beta_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxz_beta_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxz_alpha_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxz_alpha_beta_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzx_beta_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzx_alpha_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzx_alpha_beta_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxy_beta_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxy_alpha_gamma_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxy_alpha_beta_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyx_beta_gamma_0_equals_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(1.3, 0.0, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyx_alpha_gamma_0_equals_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(0.0, 1.3, 0.0);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyx_alpha_beta_0_equals_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(0.0, 0.0, 1.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r2), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyx_nearly_equals_axis_x_axis_y_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzx_nearly_equals_axis_x_axis_z_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxy_nearly_equals_axis_y_axis_x_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzy_nearly_equals_axis_y_axis_z_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxz_nearly_equals_axis_z_axis_x_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyz_nearly_equals_axis_z_axis_y_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xyz_nearly_equals_axis_x_axis_y_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_xzy_nearly_equals_axis_x_axis_z_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yxz_nearly_equals_axis_y_axis_x_and_axis_z(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_yzx_nearly_equals_axis_y_axis_z_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zxy_nearly_equals_axis_z_axis_x_and_axis_y(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
bool test_rotation_matrix_3x3_product_axes_zyx_nearly_equals_axis_z_axis_y_and_axis_x(double error)
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, action, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::z>(1.3);
    const auto r3 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::y>(-0.5);
    const auto r4 = zdm::la::RotationMatrix<double, 3, action, layout>::template coordinate_axis<zdm::Axis::x>(2.3);
    const auto r234 = zdm::la::matmul(zdm::la::matmul(r2, r3), r4);
    return is_close(std::array<double, 9>(r1), std::array<double, 9>(r234), error);
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_xy_intrinsic_equals_product_axes_yx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::y, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_xz_intrinsic_equals_product_axes_zx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::z, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_yx_intrinsic_equals_product_axes_xy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::x, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_yz_intrinsic_equals_product_axes_zy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::z, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_zx_intrinsic_equals_product_axes_xz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::x, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_zy_intrinsic_equals_product_axes_yz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::y, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_xy_extrinsic_equals_product_axes_xy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::y, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_xz_extrinsic_equals_product_axes_xz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::z, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_yx_extrinsic_equals_product_axes_yx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::x, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_yz_extrinsic_equals_product_axes_yz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::z, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_zx_extrinsic_equals_product_axes_zx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::x, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_composite_axes_zy_extrinsic_equals_product_axes_zy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::y, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_xy_intrinsic_equals_product_axes_xy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::y, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_xz_intrinsic_equals_product_axes_xz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::z, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_yx_intrinsic_equals_product_axes_yx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::x, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_yz_intrinsic_equals_product_axes_yz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::z, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_zx_intrinsic_equals_product_axes_zx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::x, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_zy_intrinsic_equals_product_axes_zy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::y, zdm::la::Chaining::intrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(1.3, -0.5);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_xy_extrinsic_equals_product_axes_yx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::x, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_xz_extrinsic_equals_product_axes_zx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::x, zdm::Axis::z, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_yx_extrinsic_equals_product_axes_xy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::x, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_yz_extrinsic_equals_product_axes_zy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::y, zdm::Axis::z, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_zx_extrinsic_equals_product_axes_xz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::x, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_composite_axes_zy_extrinsic_equals_product_axes_yz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template composite_axes<zdm::Axis::z, zdm::Axis::y, zdm::la::Chaining::extrinsic>(1.3, -0.5);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z>(-0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_xyx_intrinsic_equals_product_axes_xyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::xyx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_xzx_intrinsic_equals_product_axes_xzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::xzx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_yxy_intrinsic_equals_product_axes_yxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::yxy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_yzy_intrinsic_equals_product_axes_yzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::yzy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_zxz_intrinsic_equals_product_axes_zxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::zxz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_zyz_intrinsic_equals_product_axes_zyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::zyz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_xyx_extrinsic_equals_product_axes_xyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::xyx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_xzx_extrinsic_equals_product_axes_xzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::xzx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_yxy_extrinsic_equals_product_axes_yxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::yxy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_yzy_extrinsic_equals_product_axes_yzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::yzy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_zxz_extrinsic_equals_product_axes_zxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::zxz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_euler_angles_zyz_extrinsic_equals_product_axes_zyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_euler_angles<zdm::la::EulerConvention::zyz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_xyx_intrinsic_equals_product_axes_xyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::xyx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_xzx_intrinsic_equals_product_axes_xzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::xzx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_yxy_intrinsic_equals_product_axes_yxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::yxy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_yzy_intrinsic_equals_product_axes_yzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::yzy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_zxz_intrinsic_equals_product_axes_zxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::zxz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_zyz_intrinsic_equals_product_axes_zyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::zyz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_xyx_extrinsic_equals_product_axes_xyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::xyx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_xzx_extrinsic_equals_product_axes_xzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::xzx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_yxy_extrinsic_equals_product_axes_yxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::yxy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_yzy_extrinsic_equals_product_axes_yzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::yzy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_zxz_extrinsic_equals_product_axes_zxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::zxz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_euler_angles_zyz_extrinsic_equals_product_axes_zyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_euler_angles<zdm::la::EulerConvention::zyz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xyz_intrinsic_equals_product_axes_zyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xyz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xzy_intrinsic_equals_product_axes_yzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xzy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yxz_intrinsic_equals_product_axes_zxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yxz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yzx_intrinsic_equals_product_axes_xzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yzx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zxy_intrinsic_equals_product_axes_yxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zxy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zyx_intrinsic_equals_product_axes_xyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zyx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xyz_extrinsic_equals_product_axes_xyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xyz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xzy_extrinsic_equals_product_axes_xzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xzy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yxz_extrinsic_equals_product_axes_yxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yxz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yzx_extrinsic_equals_product_axes_yzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yzx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zxy_extrinsic_equals_product_axes_zxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zxy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zyx_extrinsic_equals_product_axes_zyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zyx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::passive, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_xyz_intrinsic_equals_product_axes_xyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xyz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_xzy_intrinsic_equals_product_axes_xzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xzy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_yxz_intrinsic_equals_product_axes_yxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yxz, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_yzx_intrinsic_equals_product_axes_yzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yzx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_zxy_intrinsic_equals_product_axes_zxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zxy, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_zyx_intrinsic_equals_product_axes_zyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zyx, zdm::la::Chaining::intrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(1.3, -0.5, 2.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_xyz_extrinsic_equals_product_axes_zyx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xyz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::y, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_xzy_extrinsic_equals_product_axes_yzx()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::xzy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::z, zdm::Axis::x>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_yxz_extrinsic_equals_product_axes_zxy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yxz, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::z, zdm::Axis::x, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_yzx_extrinsic_equals_product_axes_xzy()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::yzx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::z, zdm::Axis::y>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_zxy_extrinsic_equals_product_axes_yxz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zxy, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::y, zdm::Axis::x, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

template <zdm::la::MatrixLayout layout>
bool test_active_rotation_matrix_3x3_from_tait_bryan_angles_zyx_extrinsic_equals_product_axes_xyz()
{
    const auto r1 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template from_tait_bryan_angles<zdm::la::TaitBryanConvention::zyx, zdm::la::Chaining::extrinsic>(1.3, -0.5, 2.3);
    const auto r2 = zdm::la::RotationMatrix<double, 3, zdm::la::Action::active, layout>::template product_axes<zdm::Axis::x, zdm::Axis::y, zdm::Axis::z>(2.3, -0.5, 1.3);
    return r1 == r2;
}

void basic_linalg_tests()
{
    assert(test_matrix_row_major_layout_is_correct());
    assert(test_matrix_column_major_layout_is_correct());

    assert(test_matrix_vector_matmul_column_major_is_correct());
    assert(test_matrix_matrix_matmul_column_major_is_correct());

    assert(test_matrix_vector_matmul_row_major_is_correct());
    assert(test_matrix_matrix_matmul_row_major_is_correct());

    assert(test_vector_dot_product_is_correct());

    assert(test_vector_vector_add_is_correct());
    assert(test_vector_vector_sub_is_correct());
    assert(test_vector_vector_mul_is_correct());
    assert(test_vector_vector_div_is_correct());

    assert(test_vector_vector_add_assign_is_correct());
    assert(test_vector_vector_sub_assign_is_correct());
    assert(test_vector_vector_mul_assign_is_correct());
    assert(test_vector_vector_div_assign_is_correct());

    assert(test_vector_scalar_add_is_correct());
    assert(test_vector_scalar_sub_is_correct());
    assert(test_vector_scalar_mul_is_correct());
    assert(test_vector_scalar_div_is_correct());

    assert(test_vector_scalar_add_assign_is_correct());
    assert(test_vector_scalar_sub_assign_is_correct());
    assert(test_vector_scalar_mul_assign_is_correct());
    assert(test_vector_scalar_div_assign_is_correct());

    assert(test_scalar_vector_add_is_correct());
    assert(test_scalar_vector_sub_is_correct());
    assert(test_scalar_vector_mul_is_correct());
    assert(test_scalar_vector_div_is_correct());
}

template <zdm::la::MatrixLayout layout>
void layout_only_linalg_tests()
{
    assert(test_matrix_transpose_column_major_is_correct());
    assert(test_matrix_transpose_row_major_is_correct());

    assert((test_passive_rotation_matrix_2x2_from_angle_90_nearly_maps_pxpy_to_mypx<layout>(1.0e-15)));
    assert((test_active_rotation_matrix_2x2_from_angle_90_nearly_maps_pxpy_to_pymx<layout>(1.0e-15)));

    assert((test_passive_rotation_matrix_3x3_coordinate_axis_x_angle_90_nearly_maps_pxpypz_to_pxmzpy<layout>(1.0e-15)));
    assert((test_passive_rotation_matrix_3x3_coordinate_axis_y_angle_90_nearly_maps_pxpypz_to_pzpymx<layout>(1.0e-15)));
    assert((test_passive_rotation_matrix_3x3_coordinate_axis_z_angle_90_nearly_maps_pxpypz_to_mypxpz<layout>(1.0e-15)));

    assert((test_active_rotation_matrix_3x3_coordinate_axis_x_angle_90_nearly_maps_pxpypz_to_pxpzmy<layout>(1.0e-15)));
    assert((test_active_rotation_matrix_3x3_coordinate_axis_y_angle_90_nearly_maps_pxpypz_to_mzpypx<layout>(1.0e-15)));
    assert((test_active_rotation_matrix_3x3_coordinate_axis_z_angle_90_nearly_maps_pxpypz_to_pymxpz<layout>(1.0e-15)));

    assert((test_passive_rotation_matrix_3x3_composite_axes_xy_intrinsic_equals_product_axes_yx<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_xz_intrinsic_equals_product_axes_zx<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_yx_intrinsic_equals_product_axes_xy<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_yz_intrinsic_equals_product_axes_zy<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_zx_intrinsic_equals_product_axes_xz<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_zy_intrinsic_equals_product_axes_yz<layout>()));

    assert((test_passive_rotation_matrix_3x3_composite_axes_xy_extrinsic_equals_product_axes_xy<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_xz_extrinsic_equals_product_axes_xz<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_yx_extrinsic_equals_product_axes_yx<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_yz_extrinsic_equals_product_axes_yz<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_zx_extrinsic_equals_product_axes_zx<layout>()));
    assert((test_passive_rotation_matrix_3x3_composite_axes_zy_extrinsic_equals_product_axes_zy<layout>()));

    assert((test_active_rotation_matrix_3x3_composite_axes_xy_intrinsic_equals_product_axes_xy<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_xz_intrinsic_equals_product_axes_xz<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_yx_intrinsic_equals_product_axes_yx<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_yz_intrinsic_equals_product_axes_yz<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_zx_intrinsic_equals_product_axes_zx<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_zy_intrinsic_equals_product_axes_zy<layout>()));

    assert((test_active_rotation_matrix_3x3_composite_axes_xy_extrinsic_equals_product_axes_yx<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_xz_extrinsic_equals_product_axes_zx<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_yx_extrinsic_equals_product_axes_xy<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_yz_extrinsic_equals_product_axes_zy<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_zx_extrinsic_equals_product_axes_xz<layout>()));
    assert((test_active_rotation_matrix_3x3_composite_axes_zy_extrinsic_equals_product_axes_yz<layout>()));

    assert((test_passive_rotation_matrix_3x3_euler_angles_xyx_intrinsic_equals_product_axes_xyx<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_xzx_intrinsic_equals_product_axes_xzx<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_yxy_intrinsic_equals_product_axes_yxy<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_yzy_intrinsic_equals_product_axes_yzy<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_zxz_intrinsic_equals_product_axes_zxz<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_zyz_intrinsic_equals_product_axes_zyz<layout>()));

    assert((test_passive_rotation_matrix_3x3_euler_angles_xyx_extrinsic_equals_product_axes_xyx<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_xzx_extrinsic_equals_product_axes_xzx<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_yxy_extrinsic_equals_product_axes_yxy<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_yzy_extrinsic_equals_product_axes_yzy<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_zxz_extrinsic_equals_product_axes_zxz<layout>()));
    assert((test_passive_rotation_matrix_3x3_euler_angles_zyz_extrinsic_equals_product_axes_zyz<layout>()));

    assert((test_active_rotation_matrix_3x3_euler_angles_xyx_intrinsic_equals_product_axes_xyx<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_xzx_intrinsic_equals_product_axes_xzx<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_yxy_intrinsic_equals_product_axes_yxy<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_yzy_intrinsic_equals_product_axes_yzy<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_zxz_intrinsic_equals_product_axes_zxz<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_zyz_intrinsic_equals_product_axes_zyz<layout>()));

    assert((test_active_rotation_matrix_3x3_euler_angles_xyx_extrinsic_equals_product_axes_xyx<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_xzx_extrinsic_equals_product_axes_xzx<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_yxy_extrinsic_equals_product_axes_yxy<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_yzy_extrinsic_equals_product_axes_yzy<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_zxz_extrinsic_equals_product_axes_zxz<layout>()));
    assert((test_active_rotation_matrix_3x3_euler_angles_zyz_extrinsic_equals_product_axes_zyz<layout>()));

    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xyz_intrinsic_equals_product_axes_zyx<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xzy_intrinsic_equals_product_axes_yzx<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yxz_intrinsic_equals_product_axes_zxy<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yzx_intrinsic_equals_product_axes_xzy<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zxy_intrinsic_equals_product_axes_yxz<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zyx_intrinsic_equals_product_axes_xyz<layout>()));

    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xyz_extrinsic_equals_product_axes_xyz<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_xzy_extrinsic_equals_product_axes_xzy<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yxz_extrinsic_equals_product_axes_yxz<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_yzx_extrinsic_equals_product_axes_yzx<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zxy_extrinsic_equals_product_axes_zxy<layout>()));
    assert((test_passive_rotation_matrix_3x3_from_tait_bryan_angles_zyx_extrinsic_equals_product_axes_zyx<layout>()));

    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_xyz_intrinsic_equals_product_axes_xyz<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_xzy_intrinsic_equals_product_axes_xzy<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_yxz_intrinsic_equals_product_axes_yxz<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_yzx_intrinsic_equals_product_axes_yzx<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_zxy_intrinsic_equals_product_axes_zxy<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_zyx_intrinsic_equals_product_axes_zyx<layout>()));

    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_xyz_extrinsic_equals_product_axes_zyx<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_xzy_extrinsic_equals_product_axes_yzx<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_yxz_extrinsic_equals_product_axes_zxy<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_yzx_extrinsic_equals_product_axes_xzy<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_zxy_extrinsic_equals_product_axes_yxz<layout>()));
    assert((test_active_rotation_matrix_3x3_from_tait_bryan_angles_zyx_extrinsic_equals_product_axes_xyz<layout>()));
}

template <zdm::la::Action action, zdm::la::MatrixLayout layout>
void action_linalg_tests()
{
    assert((test_rotation_matrix_2x2_from_angle_0_is_identity<action, layout>()));

    assert((test_rotation_matrix_3x3_coordinate_axis_x_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_coordinate_axis_y_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_coordinate_axis_z_angle_0_is_identity<action, layout>()));

    assert((test_rotation_matrix_3x3_coordinate_axis_x_keeps_x_constant<action, layout>()));
    assert((test_rotation_matrix_3x3_coordinate_axis_y_keeps_y_constant<action, layout>()));
    assert((test_rotation_matrix_3x3_coordinate_axis_z_keeps_z_constant<action, layout>()));

    assert((test_rotation_matrix_3x3_coordinate_axis_x_is_nearly_orthogonal<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_coordinate_axis_y_is_nearly_orthogonal<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_coordinate_axis_z_is_nearly_orthogonal<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_axis_x_nearly_equals_coordinate_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_axis_y_nearly_equals_coordinate_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_axis_z_nearly_equals_coordinate_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_axis_is_nearly_orthogonal<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xy_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_xz_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_yx_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_yz_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_zx_angle_0_is_identity<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_zy_angle_0_is_identity<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_xy_alpha_0_equals_axis_y<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_xy_beta_0_equals_axis_x<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_xz_alpha_0_equals_axis_z<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_xz_beta_0_equals_axis_x<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_yx_alpha_0_equals_axis_x<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_yx_beta_0_equals_axis_y<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_yz_alpha_0_equals_axis_z<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_yz_beta_0_equals_axis_y<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_zx_alpha_0_equals_axis_x<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_zx_beta_0_equals_axis_z<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_zy_alpha_0_equals_axis_y<action, layout>()));
    assert((test_rotation_matrix_3x3_product_axes_zy_beta_0_equals_axis_z<action, layout>()));

    assert((test_rotation_matrix_3x3_product_axes_xy_nearly_equals_axis_x_and_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xz_nearly_equals_axis_x_and_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yx_nearly_equals_axis_y_and_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yz_nearly_equals_axis_y_and_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zx_nearly_equals_axis_z_and_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zy_nearly_equals_axis_z_and_axis_y<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xyx_beta_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xyx_alpha_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xyx_alpha_beta_0_equals_axis_x<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xzx_beta_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzx_alpha_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzx_alpha_beta_0_equals_axis_x<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_yxy_beta_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxy_alpha_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxy_alpha_beta_0_equals_axis_y<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_yzy_beta_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzy_alpha_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzy_alpha_beta_0_equals_axis_y<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_zxz_beta_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxz_alpha_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxz_alpha_beta_0_equals_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_zyz_beta_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyz_alpha_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyz_alpha_beta_0_equals_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xyz_beta_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xyz_alpha_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xyz_alpha_beta_0_equals_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xzy_beta_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzy_alpha_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzy_alpha_beta_0_equals_axis_y<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_yxz_beta_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxz_alpha_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxz_alpha_beta_0_equals_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_yzx_beta_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzx_alpha_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzx_alpha_beta_0_equals_axis_x<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_zxy_beta_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxy_alpha_gamma_0_equals_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxy_alpha_beta_0_equals_axis_y<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_zyx_beta_gamma_0_equals_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyx_alpha_gamma_0_equals_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyx_alpha_beta_0_equals_axis_x<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xyx_nearly_equals_axis_x_axis_y_and_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzx_nearly_equals_axis_x_axis_z_and_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxy_nearly_equals_axis_y_axis_x_and_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzy_nearly_equals_axis_y_axis_z_and_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxz_nearly_equals_axis_z_axis_x_and_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyz_nearly_equals_axis_z_axis_y_and_axis_z<action, layout>(1.0e-15)));

    assert((test_rotation_matrix_3x3_product_axes_xyz_nearly_equals_axis_x_axis_y_and_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_xzy_nearly_equals_axis_x_axis_z_and_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yxz_nearly_equals_axis_y_axis_x_and_axis_z<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_yzx_nearly_equals_axis_y_axis_z_and_axis_x<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zxy_nearly_equals_axis_z_axis_x_and_axis_y<action, layout>(1.0e-15)));
    assert((test_rotation_matrix_3x3_product_axes_zyx_nearly_equals_axis_z_axis_y_and_axis_x<action, layout>(1.0e-15)));
}

int main()
{
    basic_linalg_tests();

    layout_only_linalg_tests<zdm::la::MatrixLayout::column_major>();
    layout_only_linalg_tests<zdm::la::MatrixLayout::row_major>();

    action_linalg_tests<zdm::la::Action::active, zdm::la::MatrixLayout::column_major>();
    action_linalg_tests<zdm::la::Action::passive, zdm::la::MatrixLayout::column_major>();
    action_linalg_tests<zdm::la::Action::active, zdm::la::MatrixLayout::row_major>();
    action_linalg_tests<zdm::la::Action::passive, zdm::la::MatrixLayout::row_major>();
}
