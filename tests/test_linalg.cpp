#include "linalg.hpp"

#include <cassert>

bool test_matrix_row_major_layout_is_correct()
{
    zdm::Matrix<int, 2, 3, zdm::TransformAction::passive, zdm::MatrixLayout::row_major>
    matrix{{1, 2, 3, 4, 5, 6}};

    return matrix[0, 0] == 1 && matrix[0, 1] == 2 && matrix[0, 2] == 3
        && matrix[1, 0] == 4 && matrix[1, 1] == 5 && matrix[1, 2] == 6;
}

bool test_matrix_column_major_layout_is_correct()
{
    zdm::Matrix<int, 2, 3, zdm::TransformAction::passive, zdm::MatrixLayout::column_major>
    matrix{{1, 2, 3, 4, 5, 6}};

    return matrix[0, 0] == 1 && matrix[0, 1] == 3 && matrix[0, 2] == 5
        && matrix[1, 0] == 2 && matrix[1, 1] == 4 && matrix[1, 2] == 6;
}

bool test_vector_dot_product_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    return zdm::dot(v1, v2) == 1*5 + 2*7 + 3*11;
}

bool test_vector_vector_add_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    return zdm::add(v1, v2) == std::array<int, 3>{6, 9, 14};
}

bool test_vector_vector_sub_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    return zdm::sub(v1, v2) == std::array<int, 3>{-4, -5, -8};
}

bool test_vector_vector_mul_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    return zdm::mul(v1, v2) == std::array<int, 3>{5, 14, 33};
}

bool test_vector_vector_div_is_correct()
{
    std::array<int, 3> v1 = {5, 14, 33};
    std::array<int, 3> v2 = {5, 7, 11};
    return zdm::div(v1, v2) == std::array<int, 3>{1, 2, 3};
}

bool test_vector_vector_add_assign_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    zdm::add_assign(v1, v2);
    return v1 == std::array<int, 3>{6, 9, 14};
}

bool test_vector_vector_sub_assign_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    zdm::sub_assign(v1, v2);
    return v1 == std::array<int, 3>{-4, -5, -8};
}

bool test_vector_vector_mul_assign_is_correct()
{
    std::array<int, 3> v1 = {1, 2, 3};
    std::array<int, 3> v2 = {5, 7, 11};
    zdm::mul_assign(v1, v2);
    return v1 == std::array<int, 3>{5, 14, 33};
}

bool test_vector_vector_div_assign_is_correct()
{
    std::array<int, 3> v1 = {5, 14, 33};
    std::array<int, 3> v2 = {5, 7, 11};
    zdm::div_assign(v1, v2);
    return v1 == std::array<int, 3>{1, 2, 3};
}

bool test_vector_scalar_add_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::add(v, a) == std::array<int, 3>{3, 4, 5};
}

bool test_vector_scalar_sub_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::sub(v, a) == std::array<int, 3>{-1, 0, 1};
}

bool test_vector_scalar_mul_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::mul(v, a) == std::array<int, 3>{2, 4, 6};
}

bool test_vector_scalar_div_is_correct()
{
    std::array<int, 3> v = {2, 4, 6};
    int a = 2;
    return zdm::div(v, a) == std::array<int, 3>{1, 2, 3};
}

bool test_vector_scalar_add_assign_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::add_assign(v, a);
    return v == std::array<int, 3>{3, 4, 5};
}

bool test_vector_scalar_sub_assign_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::sub_assign(v, a);
    return v == std::array<int, 3>{-1, 0, 1};
}

bool test_vector_scalar_mul_assign_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    zdm::mul_assign(v, a);
    return v == std::array<int, 3>{2, 4, 6};
}

bool test_vector_scalar_div_assign_is_correct()
{
    std::array<int, 3> v = {2, 4, 6};
    int a = 2;
    zdm::div_assign(v, a);
    return v == std::array<int, 3>{1, 2, 3};
}

bool test_scalar_vector_add_is_correct()
{
    std::array<int, 3> v = {1, 2, 3};
    int a = 2;
    return zdm::add(a, v) == std::array<int, 3>{3, 4, 5};
}

bool test_scalar_vector_sub_is_correct()
{
    int a = 2;
    std::array<int, 3> v = {1, 2, 3};
    return zdm::sub(a, v) == std::array<int, 3>{1, 0, -1};
}

bool test_scalar_vector_mul_is_correct()
{
    int a = 2;
    std::array<int, 3> v = {1, 2, 3};
    return zdm::mul(a, v) == std::array<int, 3>{2, 4, 6};
}

bool test_scalar_vector_div_is_correct()
{
    int a = 6;
    std::array<int, 3> v = {6, 3, 2};
    return zdm::div(a, v) == std::array<int, 3>{1, 2, 3};
}

bool test_matrix_vector_matmul_is_correct()
{
    zdm::Matrix<int, 2, 3> m = {{1, 4, 2, 5, 3, 6}};
    std::array<int, 3> v = {1, 2, 3};
    return zdm::matmul(m, v) == std::array<int, 2>{14, 32};
}

bool test_matrix_matrix_matmul_is_correct()
{
    zdm::Matrix<int, 2, 3> m1 = {{1, 4, 2, 5, 3, 6}};
    zdm::Matrix<int, 3, 2> m2 = {{1, 3, 5, 2, 4, 6}};
    return zdm::matmul(m1, m2) == zdm::Matrix<int, 2, 2>{22, 49, 28, 64};
}

int main()
{
    assert(test_matrix_row_major_layout_is_correct());
    assert(test_matrix_column_major_layout_is_correct());

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

    assert(test_matrix_vector_matmul_is_correct());

    assert(test_matrix_matrix_matmul_is_correct());
}
