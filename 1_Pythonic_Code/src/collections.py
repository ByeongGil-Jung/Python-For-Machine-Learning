import re

"""
 Created by IntelliJ IDEA.
 Project: 1_Pythonic_Code
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-12
 Time: 오후 12:00
"""

"""

 < Collections >
>> Python 의 자료구조(data structure)를 모아둔 패키지

- list, tuple, dict 에 대한 Python Build-in 확장 자료 구조(모듈)
- 편의성, 실행 효율 등을 사용자에게 제공함
- 아래의 모듈이 존재함

from collections import deque
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple

"""


def collections():
    """ Deque """
    print("\n===============[ Deque ]===============\n")
    # stack 과 queue 를 지원하는 모듈
    # list 에 비해 효율적인 자료 저장 방식을 지원함
    # rotate, reverse 등 Linked List 의 특성을 지원함
    # 기존 list 형태의 함수를 모두 지원함
    # deque 는 기존 list 보다 효율적인 자료구조를 제공
    # 효율적 메모리 구조로 처리 속도 향상
    
    from collections import deque

    deque_list = deque()
    for i in range(5):
        deque_list.append(i)
    print(deque_list)
    deque_list.appendleft(10)
    print(deque_list)

    print("\n >> rotate")
    deque_list.rotate(2)  # deque.rotate(n) 은 n 만큼 오른쪽으로 회전
    print(deque_list)
    deque_list.rotate(2)
    print(deque_list)

    print("\n >> reversed")  # reverse 인데, 반드시 다시 deque 로 다시 묶어야 함
    print(deque_list)
    print(deque(reversed(deque_list)))

    print("\n >> extend")  # list 모양 그대로 오른쪽으로 append 할 수 있음
    print(deque_list)
    deque_list.extend([5, 6, 7])
    print(deque_list)

    print("\n >> extendleft")  # list 모양 그대로 왼쪽으로 append 할 수 있음. (거울처럼 왼쪽 value 부터 오른쪽에서 append 됨)
    print(deque_list)
    deque_list.extendleft([20, 21, 22])
    print(deque_list)

    """ OrderedDict """
    print("\n===============[ OrderedDict ]===============\n")
    # dict 와 달리, 데이터를 입력한 순서대로의 dict 를 반환함

    from collections import OrderedDict
    
    # OrderedDict 는 dict 오브젝트를 인자로 받을 수 있음
    d = OrderedDict({"a": 1, "b": 2, "c": 3})
    print(d)
    temp_dict = {"a": 10, "b": 20, "c": 30}
    d = OrderedDict(temp_dict)
    print(d)

    print("\n >> Comparing dict and OrderdDict\n")
    d = dict()  # 넣은 순서대로 정렬되지 않음
    d["x"] = 100
    d["y"] = 200
    d["z"] = 300
    d["l"] = 500
    print(d.items())

    d = OrderedDict()  # 넣은 순서대로 정렬 됨
    d["x"] = 100
    d["y"] = 200
    d["z"] = 300
    d["l"] = 500
    print(d.items())

    # dict 의 값을 value 또는 key 값으로 정렬할 때 사용 가능
    d = dict()
    d["x"] = 100
    d["y"] = 200
    d["z"] = 300
    d["l"] = 500

    # t[0] 은 dict 의 key 를 의미하며, key 를 기준으로 정렬한다. (즉, 알파벳 순)
    for k, v in OrderedDict(sorted(d.items(), key=lambda t: t[0])).items():  # 여기서 key 는 sorted 의 조건으로, 람다식 가능
        print(k, v)

    print()
    # t[1] 은 dict 의 value 를 의미하며, value 를 기준으로 정렬한다. (즉, 값 순)
    for k, v in OrderedDict(sorted(d.items(), key=lambda t: t[1])).items():
        print(k, v)

    """ Defaultdict """
    print("\n===============[ Defaultdict ]===============\n")
    # dict type 의 값에 기본값을 지정, 신규값에 default 값을 넣고자 할 때 사용

    from collections import defaultdict
    
    d = defaultdict(object)  # Default dict 를 생성
    d = defaultdict(lambda: 0)  # Default 값을 0 으로 설정
    print(d["Hello World !"])  # 만약 일반 dict 였다면 error 가 출력 됐을 것임. (값을 할당 해주지 않았기 때문)

    print("\n===============[ OrderedDict && Defaultdict ]===============\n")
    # text 에서 어떤 단어가 가장 많은지를 OrderedDict 와 defaultdict 를 이용하여 편하게 구하기

    text = "It was a cold grey day in late November. " \
           "The weather had changed overnight, " \
           "when a backing wind brought a granite sky and a mizzling rain with it, " \
           "and although it was now only a little after two o'clock in the afternoon " \
           "the pallor of a winter evening seemed to have closed upon the hills, " \
           "cloaking them in mist."
    print(text)
    print()

    text = text.lower().split()
    word_count = defaultdict(lambda: 0)
    for word in text:
        word_count[word] += 1
    print(word_count.items())

    print()
    for word, count in OrderedDict(sorted(dict(word_count).items(), key=lambda t: t[1], reverse=True)).items():
        print(word, count)

    """ Counter """
    print("\n===============[ Counter ]===============\n")
    # sequence type 의 data element 들의 값과 갯수를 dict 형태로 반환
    # dict type, keyword parameter 등도 모두 처리 가능

    from collections import Counter

    c = Counter()  # 새로운 empty Counter 인스턴스를 형성
    c = Counter("Chocolate")  # sequence 자료형의 값을 인자로 받은 Counter 인스턴스를 형성
    print(c)

    print()
    c = Counter({"red": 4, "blue": 2})
    print(c)
    print(list(c.elements()))

    print()
    c = Counter(cat=4, dog=8)
    print(c)
    print(list(c.elements()))

    """ Namedtuple """
    print("\n===============[ Namedtuple ]===============\n")
    # tuple 의 형태로 data 구조체를 저장하는 방법
    # 저장되는 data 의 변수를 사전에 지정해서 저장함
    # C 에서 구조체와 개념이 유사

    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    p = Point(11, y=22)
    print("p.x : {}, p.y : {}".format(p.x, p.y))
    print("p[0] : {}, p[1] : {}".format(p[0], p[1]))
    print("p[0] + p[1] = {}".format(p[0] + p[1]))
