# 여기서 모든 답은 한 줄로 표기 가능함


def vector_size_check(*vector_variables):
    return all(len(vector_variables[0]) == len(vector) for vector in vector_variables[1:])


"""
Other answer ::
return all(len(vector_variables[0]) == vector_len
           for vector_len in [len(vector) for vector in vector_variables[1:]])
"""


def vector_addition(*vector_variables):
    if vector_size_check(*vector_variables):
        return [sum(elements) for elements in zip(*vector_variables)]
    else:
        print("[ERROR] ArithmeticError")


def vector_subtraction(*vector_variables):
    if vector_size_check(*vector_variables):
        # 왜 elements[0] 이 zip 이 아닌, 기본 인자의 [0] value 일까 ?
        return [elements[0] * 2 - sum(elements) for elements in zip(*vector_variables)]
    else:
        print("[ERROR] ArithmeticError")


def scalar_vector_product(alpha, vector_variable):
    return [alpha * element for element in vector_variable]


def matrix_size_check(*matrix_variables):
    # 행 비교 후 열 비교
    return all(len(matrix_variables[0][0]) == len(matrix[0]) for matrix in matrix_variables[1:]) \
           and all(len(matrix_variables[0]) == len(matrix) for matrix in matrix_variables[1:])


def is_matrix_equal(*matrix_variables):
    if matrix_size_check(*matrix_variables):
        return all(matrix_variables[0] == matrix for matrix in matrix_variables[1:])
    else:
        print("[ERROR] ArithmeticError")


def matrix_addition(*matrix_variables):
    if matrix_size_check(*matrix_variables):
        return [[sum(element) for element in zip(*row)] for row in zip(*matrix_variables)]
        # 첫번째 zip 으로 묶을 땐, 각 matrix 의 같은 row 끼리 묶여진다.
        # 두번째 zip 으로 묶을 땐, 각 row 의 같은 위치에 있는 element 끼리 묶여진다.
    else:
        print("[ERROR] ArithmeticError")


def matrix_subtraction(*matrix_variables):
    if matrix_size_check(*matrix_variables):
        return [[element[0]*2 - sum(element) for element in zip(*row)] for row in zip(*matrix_variables)]
    else:
        print("[ERROR] ArithmeticError")


def matrix_transpose(matrix_variable):
    return [list(col) for col in zip(*matrix_variable)]


def scalar_matrix_product(alpha, matrix_variable):
    return [[alpha * element for element in row] for row in matrix_variable]


def is_product_availability_matrix(matrix_a, matrix_b):
    return len(matrix_a[0]) == len(matrix_b)


def matrix_product(matrix_a, matrix_b):
    if is_product_availability_matrix(matrix_a, matrix_b):
        return [[sum(r * c for r, c in zip(row, col)) for col in zip(*matrix_b)] for row in matrix_a]
    else:
        print("[ERROR] ArithmeticError")
