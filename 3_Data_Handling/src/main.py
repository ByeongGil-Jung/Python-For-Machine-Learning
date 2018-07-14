from src import numpy_lec as nl

"""
 Created by IntelliJ IDEA.
 Project: 3_Data_Handling
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-14
 Time: 오전 2:07
"""

"""

< Numpy >
- Numerical Python
- 파이썬의 고성능 과학 계산용 패키지
- Matrix 와 Vector 같은 Array 연산의 사실상의 표준
////
- 일반 list 에 비해 빠르고, 메모리 효율적
- for 문, list comprehension 등의 반복문 없이 데이터 배열에 대한 처리를 지원
- 선형대수와 관련된 다양한 기능을 제공
- 내부가 C 로 되어있기 때문에, 특히 array 연산에서 매우 빠르다.
- C, C++, 포트란 등의 언어와 통합 가능

"""

"""

[ numpy_lec.py ]


== (1) :: ndarray ==

- numpy 는 np.array 함수를 활용하여 배열을 생성함 -> ndarray
- numpy 는 하나의 데이터 타입만 넣을 수 있음 (Dynamic typing not supposed)
- (초기에 타입을 미리 설정하면, 반드시 해당 타입으로만 핸들링 가능하다.)
- C 의 Array 를 사용하여 배열을 생성하기에 매우 빠르다.
( ndarray > list comprehension > for loop 순으로 빠르다. )
- 하지만 concat 과 같이 할당을 하는 경우는 list 보다 느리다. (연속된 메모리를 찾아야 하기 때문)

< 1. reshape >
- Array 의 shape 를 변경함. (갯수는 동일)

< 2. flatten >
- 다차원의 array 를 1차원 array 로 변환

< 3. indexing >
- list 와 달리 2차원 배열에서 [0, 0] 과 같은 표기법을 제공함
- 2차원 Matrix 의 경우, 앞은 row, 뒤는 col 을 의미함

< 4. slicing >
- list 와 달리 row 와 col 부분을 나눠서 slicing 이 가능함
- Matrix 의 부분 집합을 추출할 때 유용함
- 보통 x : y 로 나타낸다. (시작과 끝 부분)
- 하지만 x : y : z 로 나타낼 수 있는데, x 는 row, y 는 col, z 는 'step' 이다.
- (즉, z 에 조건을 넣어서 step 간격으로 추출할 수 있다.)

< 5. arange >
- array 의 범위를 지정하여, 값의 list 를 생성
- 보통 np.arange(30) 과 같이 쓰지만, np,arange(0, 30, 2) 와 같이 (시작, 끝 step) 으로 생성할 수 있다.

< 6. zeros, ones, empty >
- zeros : 0 으로 가득 찬 ndarray 생성
- ones : 1 로 가득 찬 ndarray 생성
- empty : shape 만 주어지고, 비어있는 ndarray 생성
(memory initialization 이 되지 않음)

< 7. something_like >
- 기존 ndarray 의 shape 크기만큼 1, 0, 또는 empty array 를 반환

< 8. identity >
- 단위 행렬(i 행렬) 을 생성함 (n 은 row 의 갯수이다)

< 9. eye >
- 대각선의 value 가 1 인 행렬을 생성함 (k 인자의 값을 통해 시작 index 의 변경이 가능하다.)

< 10. diag >
- 대각 행렬의 값을 추출함 (k 인자의 값을 통해 시작 index 의 변경이 가능하다.)

< 11. random sampling >
- 데이터 분포에 따른 sampling 으로 array 생성
(lower boundary, upper boundary, sample size) 로 생성
m = np.random.uniform(0, 1, 10).reshape(2, 5) // Uniform Distribution
m = np.random.normal(0, 1, 10).reshape(2, 5) // Normal Distribution


== (2) :: Operation Functions ==

- numpy 는 많은 계산식을 지원한다.
- 특히, 축을 알 수 있게 해주는 "axis" 가 아주 중요하다.
(축이 늘어날 수록, 기존에 있던 축들은 한 칸씩 뒤로 밀려난다.
즉, 새로 생기는 축이 항상 0 이 된다.)

- 다양한 수학 연산자를 제공함 ...
exponential : exp, log, power, sqrt, ...
trigonometric : sin, cos, tan, arcsin, arccos, atctan, ...
hyperbolic : sinh, cosh, tanh, arcsinh, arccosh, arctang, ...

< 1. axis >
- 모든 operation function 을 수행할 때 기준이 되는 dimension 축
- (2, 3, 4) 라는 shape 가 있을 때, (2: axis=0, 3: axis=1, 4: axis=2)

< 2. sum >
- ndarray 의 element 들 간의 합을 구함
- list 의 sum 과 동일

< 3. mean & std >
- ndarray 의 element 들 간의 평균(mean) 또는 표준 편자(standard deviation) 를 구할 수 있음

< 4. concatenate >
- concatenate 를 통해 원하는 axis 를 기준으로 두 ndarray 를 결합할 수 있다.
- np.concatenate((*matrix), axis=k) 에서 axis 를 통해 concat 을 시행할 축을 결정할 수 있다.
- 하지만 numpy 에서 concat 은 느린 편이다 !
(만약, 큰 데이터가 주어진 상황이라면, numpy 의 concatenate 를 사용하는 것은 비효율적이다. -> list 가 더 효율적이다.)


== (3) :: Array Operations ==

- numpy 는 array 간의 기본적인 사칙 연산을 지원함
- shape 가 서로 다른 배열 간 연산을 지원하는 기능인 broadcasting 이 중요

< 1. basic operations >
- matrix 의 shape 가 같다면, 기본적인 사칙 연산을 지원함

< 2. dot product >
- Matrix 의 기본 product 연산 (행렬의 곱)
m = np.dot(matrix_a, matrix_b)

< 3. transpose >
- Matrix 의 전치 행렬 (transpose)
m = np.transpose(matrix)

< 4. broadcasting >
- shape 이 서로 다른 배열 간의 연산을 지원함
- Scalar - Vector 외에도, Vector - Matrix 간의 연산도 지원


== (4) :: Comparisons ==

< 1. all & any >
- ndarray 의 데이터 전부(and) 또는 일부(or) 가 조건에 만족하는지의 여부 반환

< 2. where >
- 조건에 만족하는 index 값을 추출한다
np.where(condition, True, False)

np.where(m > 10, 7, 8)  // 10 보다 큰 index 에선 7, 작은 index 에선 8 추출
np.where(m > 10)  // 10 보다 큰 element 가 존재하는 index 반환

< 3. argmax & argmin >
- array 내 최대값 또는 최소값의 index 를 반환 (axis 를 기준으로 반환한다.)
m = np.argmax(m, axis=1)
m = np.argmin(m, axis=0)


== (5) :: Boolean & Fancy Index ==

< 1. boolean index >
- numpy 는 특정 배열에서 특수한 조건에서의 추출한 값을 다시 배열 형태로 추출할 수 있음
- Comparison Operation 함수들도 모두 사용 가능
m2 = m[m > 3]  // 조건이 True (m > 3) 인 index 의 element 만을 추출

< 2. fancy index (take) >
- numpy 는 배열 index 의 value 를 이용하여 값을 추출할 수 있음
- 이 때, index 로 사용될 배열은 반드시 integer 로 선언되어야 한다.
- Matrix 형태의 데이터에도 사용 가능하다.
m = np.array([2, 4, 6, 8], dtype=np.float64)
arr = np.array([0, 0, 1, 3, 2, 1], dtype=int)
print(np.take(m, arr))
print(m[arr])
> [2, 2, 4, 8, 6, 4]

"""


def main():
    print("\n#############################[ 1. numpy_lec.py ]#############################")
    nl.numpy_lec()


if __name__ == "__main__":
    main()
