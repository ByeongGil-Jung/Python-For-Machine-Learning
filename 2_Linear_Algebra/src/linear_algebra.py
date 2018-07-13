"""
 Created by IntelliJ IDEA.
 Project: 2_Linear_Algebra
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-13
 Time: 오전 2:28
"""


def linear_algebra():
    """ Vector """
    print("\n===============[ Vector ]===============\n")
    # Vector 를 Python 으로 표시하는 다양한 방법 존재
    # 최선의 방법은 없음
    # 값의 변경 유무, 속성값 유무에 따라 선택할 수 있음
    # 예제에선 기본적으로 vector 를 list 로 연산할 것임
    # zip 을 사용해서 vector 를 간단하게 계산할 수 있다.

    vector_a = [1, 2, 10]  # list 로 표현했을 경우
    vector_b = (1, 2, 10)  # tuple 로 표현했을 경우
    vector_c = {"x": 1, "y": 2, "z": 10}  # dict 로 표현했을 경우
    print("Vector of list : {}".format(vector_a))
    print("Vector of tuple : {}".format(vector_b))
    print("Vector of dict : {}".format(vector_c))

    # vector 의 덧셈
    print("\n >> Addition of vector")
    u = [2, 2]
    v = [2, 3]
    z = [3, 5]
    print("Vector u : {}".format(u))
    print("Vector v : {}".format(v))
    print("Vector z : {}".format(z))
    result = [sum(t) for t in zip(u, v, z)]
    print(result)  # [7, 10]
    
    # scalar-vector 곱
    print("\n >> Scalar-Vector product")
    u = [1, 2, 3]
    v = [4, 4, 4]
    scalar = 2
    print("Vector u : {}".format(u))
    print("Vector v : {}".format(v))
    print("Scalar : {}".format(scalar))
    result = [scalar * sum(t) for t in zip(u, v)]
    print(result)

    """ Matrix """
    print("\n===============[ Matrix ]===============\n")
    # Matrix 역시 Python 으로 표시하는 다양한 방법 존재
    # 특히 dict 로 표현할 때는 무궁무진한 방법 존재
    # 예제에선 기본적으로 2-Dimensional-List 형태로 표현함
    # [[1번째 row], [2번째 row], [3번째 row]]
    
    matrix_a = [[3, 6], [4, 5]]  # list 로 표현했을 경우
    matrix_b = [(3, 6), (4, 5)]  # tuple 로 표현했을 경우
    matrix_c = {(0, 0): 3, (0, 1): 6, (1, 0): 4, (1, 1): 5}  # dict 로 표현했을 경우
    print("Matrix of list : {}".format(matrix_a))
    print("Matrix of tuple : {}".format(matrix_b))
    print("Matrix of dict : {}".format(matrix_c))

    # Matrix 의 덧셈
    print("\n >> Addition of matrix")
    matrix_a = [[3, 6], [4, 5]]
    matrix_b = [[5, 8], [6, 7]]
    print("Matrix a : {}".format(matrix_a))
    print("Matrix b : {}".format(matrix_b))
    
    temp = [t for t in zip(matrix_a, matrix_b)]  # 두 개의 tuple 형태로 matrix 가 형성됨 -> 활용하기 위해선 *temp 로 unpacking 필요
    print(temp)
    
    result = [[sum(row) for row in zip(*t)] for t in zip(matrix_a, matrix_b)]
    print(result)

    # scalar-matrix 곱
    print("\n >> Scalar-Matrix product")
    matrix_a = [[3, 6], [4, 5]]
    scalar = 4
    print("Matrix a : {}".format(matrix_a))
    print("Scalar : {}".format(scalar))

    temp = [t for t in matrix_a]
    print(temp)

    result = [[scalar * element for element in t] for t in matrix_a]
    print(result)

    # matrix transpose
    print("\n >> Transpose of matrix")
    matrix_a = [[1, 2, 3], [4, 5, 6]]
    print("Matrix a : {}".format(matrix_a))

    temp = [t for t in zip(matrix_a)]  # ([1, 2, 3],), ([4, 5, 6],) 인 두 tuple 로 나뉘어짐 -> 따라서 *matrix_a 으로 unpacking 해줘야 함
    print(temp)
    temp = [t for t in zip(*matrix_a)]  # [1, 2, 3], [4, 5, 6] 으로 unpacking 된 상태로 zip 함수 실행 -> (1, 4), (2, 5), (3, 6)
    print(temp)

    result = [[element for element in t] for t in zip(*matrix_a)]
    print(result)

    # matrix 곱셈
    print("\n >> Matrix product")
    matrix_a = [[1, 1, 2], [2, 1, 1]]
    matrix_b = [[1, 1], [2, 1], [1, 3]]
    print("Matrix a : {}".format(matrix_a))
    print("Matrix b : {}".format(matrix_b))

    temp = [row_a for row_a in matrix_a]
    print(temp)

    temp2 = [col_b for col_b in zip(*matrix_b)]
    print(temp2)

    result = [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*matrix_b)] for row_a in matrix_a]
    print(result)
