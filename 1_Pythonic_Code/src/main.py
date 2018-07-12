from src import basic_code as bc
from src import collections as c

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

"""

"""

[ basic_code.py ]

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

"""

[ collections.py ]

< 1. Deque >
- stack 과 queue 를 지원하는 모듈
- list 에 비해 효율적인 자료 저장 방식을 지원함
- rotate, reverse 등 Linked List 의 특성을 지원함
- 기존 list 형태의 함수를 모두 지원함
- deque 는 기존 list 보다 효율적인 자료구조를 제공
- 효율적 메모리 구조로 처리 속도 향상

< 2. OrderedDict >
- dict 와 달리, OrderedDict 는 데이터를 입력한 순서대로의 dict 를 반환함
- OrderedDict 는 dict 오브젝트를 인자로 받을 수 있음

< 3. DefaultDict >
- dict type 의 값에 기본값을 지정, 신규값에 default 값을 넣고자 할 때 사용

< 4. Counter >
- sequence type 의 data element 들의 값과 갯수를 dict 형태로 변환
- dict type, keyword parameter 등도 모두 처리 가능

< 5. Namedtuple >
- tuple 의 형태로 data 구조체를 저장하는 방법
- 저장되는 data 의 변수를 사전에 지정해서 저장함
- C 에서 구조체와 개념이 유사

"""


def main():
    print("\n#############################[ 1. basic code.py ]#############################")
    bc.split_join()
    bc.list_comprehensions()
    bc.enumerate_zip()
    bc.lambda_map_reduce()
    bc.asterisk()

    print("\n#############################[ 2. collections.py ]#############################")
    c.collections()


if __name__ == "__main__":
    main()
