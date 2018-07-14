import numpy as np

"""
 Created by IntelliJ IDEA.
 Project: 3_Data_Handling
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-14
 Time: 오전 2:12
"""


def numpy_lec():
    """ Ndarray """
    print("\n===============[ Ndarray ]===============")
    # numpy 는 np.array 함수를 활용하여 배열을 생성함 -> ndarray
    # numpy 는 하나의 데이터 type 만 배열에 넣을 수 있음
    # list 와의 가장 큰 차이점은, Dynamic typing not supported 라는 점
    # (초기에 타입을 미리 선언하면, 반드시 해당 타입으로만 핸들링 가능하다.)
    # C 의 Array 를 사용하여 배열을 생성함
    # ( numpy > list comprehension > for loop 순으로 속도가 빠르다.)
    # 하지만 concat 같이 할당을 하는 경우는 속도가 list 보다 느리다 !

    test_array = np.array([1, 4, 5, "8"], dtype=np.float64)  # str 값을 넣어도 해당 type 으로 형변환됨
    print(test_array)  # [1. 4. 5. 8.]
    print(type(test_array[3]))  # numpy.float64
    print(test_array.shape)  # (4,)
    print(test_array.size)  # 4 (총 element 의 갯수)
    print(test_array.nbytes)  # 32 (8 bytes * 4)

    print("\n >> reshape\n")
    # Array 의 shape 를 변경함 (element 의 갯수는 동일)

    test_matrix = [[1, 2, 3], [4, 5, 6]]
    print(np.array(test_matrix))
    print(">", np.array(test_matrix).shape, type(np.array(test_matrix)))
    print("\nreshape (3, 2, 1)")
    print(np.array(test_matrix).reshape(3, 2, 1))  # 순서대로 z, y, x 축 값이라 생각
    print("\nreshape (-1, 2)")
    print(np.array(test_matrix).reshape(-1, 2))  # 앞에 -1 을 붙이면, size 를 기반으로 col 의 갯수를 나눈다
    print("\nreshape (6,)")
    print(np.array(test_matrix).reshape(6, ))

    print("\n >> flatten")
    # 다차원 array 를 1차원 array 로 변환

    test_matrix = [[[1, 2, 3, 4], [1, 2, 5, 8]], [[1, 2, 3, 4], [1, 2, 5, 8]]]
    print(np.array(test_matrix))
    print(">", np.array(test_matrix).shape)
    print()
    print(np.array(test_matrix).flatten())

    print("\n >> indexing\n")
    # list 와 달리 2차원 배열에서 [0, 0] 과 같은 표기법을 제공함
    # 2차원 Matrix 일 경우, 앞은 row, 뒤는 col 을 의미함

    test_matrix = np.array([[1, 2, 3], [4.5, 5, 6]], dtype=np.int)
    print(test_matrix)
    print(test_matrix[1, 2])  # 6
    print(test_matrix[1][2])  # 6

    test_matrix[1, 2] = 16
    test_matrix[0][2] = 30
    print(test_matrix)

    print("\n >> slicing\n")
    # list 와 달리 row 와 col 부분을 나눠서 slicing 이 가능함
    # Matrix 의 부분 집합을 추출할 때 유용함

    # 보통 x : y 로 나타낸다.
    # 하지만 x: y: z 로 나타낼 수 있는데, x 는 row, y 는 col, z 는 'step' 이다.
    # (즉, z 에 조건을 넣어서 step 으로 추출할 수 있다.)

    m = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int)
    print(m)
    print()
    print(m[:, 2:])  # row 전체, 2 <= col
    print(m[1, 1:3])  # 1 row, 1 <= col < 3
    print(m[1:2])  # 1 row 전체

    print()
    m = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=np.int)
    print(m)
    print()
    print(m[:, ::2])  # col 에서 index 2 씩 step
    print(m[::2, ::3])  # row 에서 index 2 씩 step, col 에서 index 3 씩 step

    print("\n >> arange\n")
    # array 의 범위를 지정하여, 값의 list 를 생성
    # 보통 np.arange(30) 처럼 쓰지만, np.arange(0, 30, 2) 와 같이, (시작, 끝, step) 등을 표기할 수 있다.

    m = np.arange(30, dtype=np.int)
    print(m)
    m = np.arange(0, 30, 2, dtype=np.int)
    print(m)
    m = np.arange(30, dtype=np.int).reshape(5, 6)
    print(m)

    print("\n >> zeros, ones, empty\n")
    # zeros :: 0 으로 가득 찬 ndarray 생성
    # ones :: 1 로 가득 찬 ndarray 생성
    # empty :: shape 만 주어지고, 비어있는 ndarray 생성
    # (memory initialization 이 되지 않음)

    m = np.zeros(shape=(10,), dtype=np.int)
    print(m)
    m = np.zeros((2, 5), dtype=np.int)
    print(m)

    print()
    m = np.ones(shape=(15,), dtype=np.int)
    print(m)
    m = np.ones((3, 5), dtype=np.int)
    print(m)

    print()
    m = np.empty(shape=(8,))
    print(m)
    m = np.empty((3, 5))
    print(m)

    print("\n >> something_like\n")
    # 기존 ndarray 의 shpae 크기 만큼 1, 0, 또는 empty array 를 반환

    m = np.arange(30, dtype=np.int).reshape(5, 6)
    print(m)
    print(np.zeros_like(m))
    print(np.ones_like(m))

    print("\n >> identity\n")
    # 단위 행렬(i 행렬) 을 생성함 (n 은 row 의 갯수이다.)

    i = np.identity(n=3, dtype=np.int)
    print(i)
    i = np.identity(5, dtype=np.int)
    print(i)

    print("\n >> eye\n")
    # 대각선이 1인 행렬을 생성함 (k 인자를 통해 시작 index 의 변경이 가능하다.)

    e = np.eye(N=3, M=5, dtype=np.int)
    print(e)
    e = np.eye(3, 5, k=2, dtype=np.int)
    print(e)
    e = np.eye(5, k=2, dtype=np.int)
    print(e)

    print("\n >> diag\n")
    # 대각 행렬의 값을 추출함 (k 인자를 통해 시작 index 의 변경이 가능하다.)

    m = np.arange(9).reshape(3, 3)
    print(m)
    print(np.diag(m))  # [0, 4, 8]
    print(np.diag(m, k=1))  # [1, 5]

    print("\n >> random sampling\n")
    # 데이터 분포에 따른 sampling 으로 array 생성
    # (lower boundary, upper boundary, sample size) 로 생성

    print("Uniform Distribution")
    m = np.random.uniform(0, 1, 10).reshape(2, 5)
    print(m)

    print("\nNormal Distribution")
    m = np.random.normal(0, 1, 10).reshape(2, 5)
    print(m)

    """ Operation Functions """
    print("\n===============[ Operation Functions ]===============")
    # numpy 는 많은 계산식을 지원한다.
    # 특히, 축을 알 수 있게 해주는 "axis" 가 아주 중요하다.
    # (축이 늘어날 수록, 기존에 있던 축들은 한 칸씩 뒤로 밀려난다.
    # 즉, 새로 생기는 축이 항상 0 이 된다.)

    # 다양한 수학 연산자를 제공함
    # exponential: exp, log, power, sqrt, ...
    # trigonometric: sin, cos, tan, arcsin, arccos, arctan, ...
    # hyperbolic: sinh, cosh, tanh, arcsinh, arccosh, arctanh, ...

    print("\n >> axis\n")
    # 모든 operation function 을 수행 할 때 기준이 되는 dimension 축
    # (2, 3, 4) 라는 shape 가 있을 때, (2: axis=0, 3: axis=1, 4: axis=2)

    m = np.arange(1, 13, dtype=np.int).reshape(3, 4)
    print(m)
    print(np.sum(m))  # 모든 element 의 합을 구함
    print(np.sum(m, axis=0))  # col 을 기준으로 잡음
    print(np.sum(m, axis=1))  # row 를 기준으로 잡음

    print("\n >> sum\n")
    # ndarray 의 element 들 간의 합을 구함
    # list 의 sum 기능과 동일

    m = np.arange(1, 11, dtype=np.int)
    print(m)
    print(np.sum(m, dtype=np.float64))  # 모든 element 들의 합

    print("\n >> mean & std\n")
    # ndarray 의 element 들 간의 평균(mean) 또는 표준 편차(standard deviation) 을 구할 수 있음

    m = np.arange(1, 13).reshape(3, 4)
    print(m)
    print(np.mean(m))  # 전체 평균 (6.5)
    print(np.mean(m, axis=0))  # axis=0 의 평균
    print(np.std(m, axis=1))  # axis=1 의 표준 편차

    """ Concatenate """
    print("\n===============[ Concatenate ]===============")
    # concatenate 를 통해 원하는 axis 를 기준으로 결합할 수 있다.

    print("\n >> concatenate\n")
    # numpy array 를 합치는 함수
    # np.concatenate((*matrix), axis=k) 를 통해 concat 을 시행할 축을 결정할 수 있다.
    # 하지만 numpy 에서 concat 은 느린 편이다 !
    # (만약, 큰 데이터가 주어진 상황이라면, numpy 의 concat 은 비효율적이다. -> list 가 더 효율적이다.)

    matrix_a = np.array([[1, 2, 3]])
    matrix_b = np.array([[2, 3, 4]])
    print(matrix_a, matrix_b)
    print()
    print(np.concatenate((matrix_a, matrix_b), axis=0))
    print(np.concatenate((matrix_a, matrix_b), axis=1))

    print()
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6]])
    print(matrix_a, matrix_b)
    print(np.concatenate((matrix_a, matrix_b), axis=0))
    # 만약 axis 의 feature 갯수가 다르다면, matrix_b 와 같이 transpose 할 것
    print(np.concatenate((matrix_a, np.transpose(matrix_b)), axis=1))

    """ Array Operations """
    print("\n===============[ Array Operations ]===============")
    # numpy 는 array 간의 기본적인 사칙 연산을 지원함
    # shape 가 서로 다른 배열 간 연산을 지원하는 기능인 broadcasting 이 중요

    print("\n >> basic operations\n")
    # matrix 의 shape 이 같다면, 기본적인 사칙 연산을 지원함

    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    print(m)
    print()
    print(m + m)  # Matrix + Matrix 연산
    print(m - m)  # Matrix - Matrix 연산
    print(m * m)  # Matrix 내 같은 element 위치의 값 끼치 연산

    print("\n >> dot product\n")
    # Matrix 의 기본 연산 (행렬의 곱)
    # dot 함수 사용

    matrix_a = np.arange(1, 7, dtype=np.int).reshape(2, 3)
    matrix_b = np.arange(7, 13, dtype=np.int).reshape(3, 2)
    print(matrix_a)
    print(matrix_b)
    print()
    print(np.dot(matrix_a, matrix_b))

    print("\n >> transpose\n")
    # Matrix 의 전치 행렬 (transpose)

    m = np.arange(1, 7, dtype=np.int).reshape(2, 3)
    print(m)
    print(np.transpose(m))
    print(np.transpose(m).dot(m))  # Matrix 간 곱셈

    print("\n >> broadcasting\n")
    # shape 이 서로 다른 배열 간의 연산을 지원함
    # Scalar - Vector 외에도, Vector - Matrix 간의 연산도 지원

    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    scalar = 3

    print(m, scalar)
    print()
    print(m + scalar)  # Matrix - Scalar 덧셈
    print(m - scalar)  # Matrix - Scalar 뺄셈
    print(m * scalar)  # Matrix - Scalar 곱셈
    print(m / scalar)  # Matrix - Scalar 나눗셈
    print(m % scalar)  # Matrix - Scalar 나머지
    print(m ** scalar)  # Matrix - Scalar 제곱

    # Matrix - Vector 간의 연산도 지원
    print()
    m = np.arange(1, 13, dtype=np.int).reshape(4, 3)
    vector = [0, 10, 100]
    print(m)
    print(vector)
    print()
    print(m + vector)  # Matrix - Vector 덧셈 (이하 동일)

    """ Comparisons """
    print("\n===============[ Comparisons ]===============")

    print("\n >> all & any\n")
    # array 의 데이터 전부(and) 또는 일부(or) 가 조건에 만족하는지의 여부 반환

    m = np.arange(10, dtype=np.int)

    print(m)
    print(m > 5)
    print(np.all(m > 5))
    print(np.any(m > 5))

    print("\n >> where\n")
    # 조건에 만족하는 index 값을 추출한다.
    # np.where(condition, True, False)

    m = np.arange(1, 13, dtype=np.int)
    print(m)
    print(np.where(m > 10, 7, 8))  # 10 보다 큰 index 에선 7, 작은 index 에선 8 추출
    print(np.where(m > 10))  # 10 보다 큰 element 가 존재하는 index 반환. (여기선 10, 11 이다.)

    print("\n >> argmax & argmin\n")
    # array 내 최대값 또는 최소값의 index 를 반환함
    # axis 기준으로 반환

    m = np.array([1, 2, 4, 5, 8, 78, 23, 3], dtype=np.int)
    print(m)
    print(np.argmax(m))  # 5 (m[5] == 78)
    print(np.argmin(m))  # 0 (m[0] == 1)

    print()
    m = np.array([[1, 2, 4, 7], [9, 88, 6, 45], [9, 76, 3, 4]], dtype=np.int)
    print(m)
    print(np.argmax(m, axis=1))  # [3, 1, 1] (m[0][3] == 7 // m[1][1] == 88 // m[2][1] == 76)
    print(np.argmin(m, axis=0))  # [0, 0, 2, 2] (m[0][0] == 1 // m[0][1] == 2 // m[2][2] == 3 // m[2][3] == 4)

    """ Boolean & Fancy Index """
    print("\n===============[ Boolean & Fancy Index ]===============")

    print("\n >> boolean index\n")
    # numpy 는 특정 배열에서 특수한 조건에서의 추출한 값을 다시 배열 형태로 추출할 수 있음
    # Comparison Operation 함수들도 모두 사용가능

    m = np.array([1, 4, 0, 2, 3, 8, 9, 7], dtype=np.float64)
    print(m)
    print(m > 3)  # [False, True, False, False, False, True, True, True] // index : 1, 5, 6, 7
    print(m[m > 3])  # 조건이 True 인 index 의 element 만 추출

    condition = m < 3
    print(m[condition])  # [1, 0, 2] // index : 0, 2, 3

    # 이를 통해 원하는 matrix 의 boolean (혹은 0/1) 값을 구할 수 있다
    print()
    big_m = np.array(
        [[12, 13, 14, 12, 16, 14, 11, 10, 9],
         [11, 14, 12, 15, 15, 16, 10, 12, 11],
         [10, 12, 12, 15, 14, 16, 10, 12, 12],
         [9, 11, 16, 15, 14, 16, 15, 12, 10],
         [12, 11, 16, 14, 10, 12, 16, 12, 13],
         [10, 15, 16, 14, 14, 14, 16, 15, 12],
         [13, 17, 14, 10, 14, 11, 14, 15, 10],
         [10, 16, 12, 14, 11, 12, 14, 18, 11],
         [10, 19, 12, 14, 11, 12, 14, 18, 10]], dtype=np.int)

    condition = big_m < 15
    print(big_m)
    print(condition)
    print(condition.astype(np.int))

    print("\n >> fancy index (take)\n")
    # numpy 는 배열 index 의 value 를 이용하여 값을 추출할 수 있음
    # 이 때, index 로 이용될 배열은 반드시 integer 로 선언되어야 한다.
    # Matrix 형태의 데이터도 사용 가능하다

    m = np.array([2, 4, 6, 8], dtype=np.float64)  # 추출하고자 하는 배열
    arr = np.array([0, 0, 1, 3, 2, 1], dtype=np.int)  # index 로 이용될 배열
    print(np.take(m, arr))  # [2, 2, 4, 8, 6, 4]
    print(m[arr])  # take 와 동일하지만, 가독성상 좋지 않다.

    # Matrix 형태로 사용하는 방식
    print()
    m = np.array([[1, 4], [9, 16]], np.float64)
    arr1 = np.array([0, 0, 1, 1, 0], np.int)
    arr2 = np.array([0, 1, 1, 1, 1], np.int)

    print(m)
    print(m[arr1, arr2])  # arr1 을 row index, arr2 을 col index 로 변환하여 표시함
