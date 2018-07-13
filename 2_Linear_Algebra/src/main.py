from src import linear_algebra as la
from src import linear_algebra_practice as lap

"""
 Created by IntelliJ IDEA.
 Project: 2_Linear_Algebra
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-13
 Time: 오전 2:25
"""


def main():
    print("\n#############################[ 1. linear_algebra.py ]#############################")
    la.linear_algebra()

    print("\n#############################[ 2. linear_algebra_practice.py ]#############################")
    print("\n >> 1. vector_size_check() ::")

    print(lap.vector_size_check([1, 2, 3], [2, 3, 4], [5, 6, 7]))  # Expected value: True
    print(lap.vector_size_check([1, 3], [2, 4], [6, 7]))  # Expected value: True
    print(lap.vector_size_check([1, 3, 4], [4], [6, 7]))  # Expected value: False

    print("\n >> 2. vector_addition() ::")

    print(lap.vector_addition([1, 3], [2, 4], [6, 7]))  # Expected value: [9, 14]
    print(lap.vector_addition([1, 5], [10, 4], [4, 7]))  # Expected value: [15, 16]
    print(lap.vector_addition([1, 3, 4], [4], [6, 7]))  # Expected value: ArithmeticError

    print("\n >> 3. vector_subtraction() ::")

    print([elements[0] for elements in zip(*([1, 3], [2, 4]))])  # 왜 elements[0] 이 zip 이 아닌, 기본 인자의 [0] value 일까 ?
    print(lap.vector_subtraction([1, 3], [2, 4]))  # Expected value: [-1, -1]
    print(lap.vector_subtraction([1, 5], [10, 4], [4, 7]))  # Expected value: [-13, -6]

    print("\n >> 4. scalar_vector_product() ::")

    print(lap.scalar_vector_product(5, [1, 2, 3]))  # Expected value: [5, 10, 15]
    print(lap.scalar_vector_product(3, [2, 2]))  # Expected value: [6, 6]
    print(lap.scalar_vector_product(4, [1]))  # Expected value: [4]

    print("\n >> 5. matrix_size_check() ::")

    matrix_x = [[2, 2], [2, 2], [2, 2]]
    matrix_y = [[2, 5], [2, 1]]
    matrix_z = [[2, 4], [5, 3]]
    matrix_w = [[2, 5], [1, 1], [2, 2]]

    print(lap.matrix_size_check(matrix_x, matrix_y, matrix_z))  # Expected value: False
    print(lap.matrix_size_check(matrix_y, matrix_z))  # Expected value: True
    print(lap.matrix_size_check(matrix_x, matrix_w))  # Expected value: True

    print("\n >> 6. is_matrix_equal() ::")

    matrix_x = [[2, 2], [2, 2]]
    matrix_y = [[2, 5], [2, 1]]

    print(lap.is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y))  # Expected value: False
    print(lap.is_matrix_equal(matrix_x, matrix_x))  # Expected value: True

    print("\n >> 7. matrix_addition() ::")

    matrix_x = [[2, 2], [2, 2]]
    matrix_y = [[2, 5], [2, 1]]
    matrix_z = [[2, 4], [5, 3]]

    print(lap.matrix_addition(matrix_x, matrix_y))  # Expected value: [[4, 7], [4, 3]]
    print(lap.matrix_addition(matrix_x, matrix_y, matrix_z))  # Expected value: [[6, 11], [9, 6]]

    print("\n >> 8. matrix_subtraction() ::")

    matrix_x = [[2, 2], [2, 2]]
    matrix_y = [[2, 5], [2, 1]]
    matrix_z = [[2, 4], [5, 3]]

    print(lap.matrix_subtraction(matrix_x, matrix_y))  # Expected value: [[0, -3], [0, 1]]
    print(lap.matrix_subtraction(matrix_x, matrix_y, matrix_z))  # Expected value: [[-2, -7], [-5, -2]]

    print("\n >> 9. matrix_transpose() ::")

    matrix_w = [[2, 5], [1, 1], [2, 2]]  # Expected value: [[2, 1, 2], [5, 1, 2]]
    print(lap.matrix_transpose(matrix_w))

    print("\n >> 10. scalar_matrix_product() ::")

    matrix_x = [[2, 2], [2, 2], [2, 2]]
    matrix_y = [[2, 5], [2, 1]]
    matrix_z = [[2, 4], [5, 3]]
    matrix_w = [[2, 5], [1, 1], [2, 2]]

    print(lap.scalar_matrix_product(3, matrix_x))  # Expected value: [[6, 6], [6, 6], [6, 6]]
    print(lap.scalar_matrix_product(2, matrix_y))  # Expected value: [[4, 10], [4, 2]]
    print(lap.scalar_matrix_product(4, matrix_z))  # Expected value: [[8, 16], [20, 12]]
    print(lap.scalar_matrix_product(3, matrix_w))  # Expected value: [[6, 15], [3, 3], [6, 6]]

    print("\n >> 11. is_product_availability_matrix() ::")

    matrix_x = [[2, 5], [1, 1]]
    matrix_y = [[1, 1, 2], [2, 1, 1]]
    matrix_z = [[2, 4], [5, 3], [1, 3]]

    print(lap.is_product_availability_matrix(matrix_y, matrix_z))  # Expected value: True
    print(lap.is_product_availability_matrix(matrix_z, matrix_x))  # Expected value: True
    print(lap.is_product_availability_matrix(matrix_z, matrix_w))  # Expected value: False // matrix_w 가 없습니다
    print(lap.is_product_availability_matrix(matrix_x, matrix_x))  # Expected value: True

    print("\n >> 12. matrix_product() ::")

    matrix_x = [[2, 5], [1, 1]]
    matrix_y = [[1, 1, 2], [2, 1, 1]]
    matrix_z = [[2, 4], [5, 3], [1, 3]]

    print(lap.matrix_product(matrix_y, matrix_z))  # Expected value: [[9, 13], [10, 14]]
    print(lap.matrix_product(matrix_z, matrix_x))  # Expected value: [[8, 14], [13, 28], [5, 8]]
    print(lap.matrix_product(matrix_x, matrix_x))  # Expected value: [[9, 15], [3, 6]]
    print(lap.matrix_product(matrix_z, matrix_w))  # Expected value: False


if __name__ == "__main__":
    main()
