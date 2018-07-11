from src import basic_code as bc

"""
 Created by IntelliJ IDEA.
 Project: 1_Pythonic_Code
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-11
 Time: 오후 4:35
"""

"""

< Pythonic Code >
- Python 개발자들과 커뮤니케이션 가능
- 효율

< 1. Split & Join >
- pass

< 2. List Comprehensions >
- 기본 List 를 사용하여 간단히 다른 List 를 만드는 기법
- 포괄적인 List, 포함되는 List 라는 의미로 사용됨
- 파이썬에서 가장 많이 사용되는 기법 중 하나
- 일반적으로 for + append 보다 속도가 빠름

< 3. Enumerate >
- list 의 element 를 추출할 때 번호를 붙여서 추출

< 4. Zip >
- 두 개의 list 의 값을 병렬적으로 추출

>> Enumerate 와 Zip 을 동시에 써서, 순서쌍을 나타낼 수 있음

< 5. Lambda >
- 함수 이름 없이, 함수처럼 쓸수 있는 익명함수
- lambda 에도 필터를 넣을 수 있지만, 반드시 else 문을 기입해줘야 한다.

< 6. Map >
- Sequence 자료형의 각 element 에 동일한 function 을 적용함
- Python3 에선 반드시 list 나 tuple 같은 sequence 자료형으로 반환을 해줘야 추출 가능하다.
(하지만 iterator 로 하나하나씩 호출 가능하다.)

< 7. Reduce >
- map 과 달리 sequence 자료형에 똑같은 함수를 적용해서 앞의 element 부터 누적하여 계산
- 'from functools import reduce' 로 import 해줘야 함

< 8. Asterisk >
- '*' 를 의미한다.
- Python 에선 단순 곱셈, 제곱 연산, 가변 인자 활용, unpacking 등 다양하게 활용된다.
(unpcking 시,
 *list :: list 를 unpacking 한다.
 **dict :: dict 를 unpacking 한다.)

"""


def main():
    bc.split_join()
    bc.list_comprehensions()
    bc.enumerate_zip()
    bc.lambda_map_reduce()
    bc.asterisk()


if __name__ == "__main__":
    main()