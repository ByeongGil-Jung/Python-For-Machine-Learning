from functools import reduce

"""
 Created by IntelliJ IDEA.
 Project: 1_Pythonic_Code
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-11
 Time: 오후 5:10
"""


def split_join():
    """ Split """
    print("\n===============[ Split ]===============")

    # String Type 이 값을 나눠서 list 형태로 변환
    print("\n >> Splitting String values to List value\n")
    items = "zero one two three".split()  # 빈칸을 기준으로 문자열 나누기
    print(items)

    example = "python,jquery,javascript"
    example_splitted = example.split(",")  # ',' 을 기준으로 문자열 나누기
    print(example_splitted)

    (a, b, c) = example.split(",")  # list 에 있는 각 값을 a, b, c 변수로 unpacking
    print(a)
    print(a, b, c)

    example = "cs50.gachon.edu"
    (subdomain, domain, tld) = example.split(".")  # "." 을 기준으로 문자열 나누기 (unpacking)
    print(subdomain, domain, tld)

    """ Join """
    print("\n===============[ Join ]===============")

    # String list 를 합쳐 하나의 String 으로 반환할 때 사용
    print("\n >> Combining several String list return to one String\n")
    colors = ["red", "blue", "green", "yellow"]
    result = "".join(colors)
    print(result)
    result = " ".join(colors)  # 연결 시 빈 칸으로 연결
    print(result)
    result = ", ".join(colors)  # 연결 시 ", " 으로 연결
    print(result)
    result = "-".join(colors)  # 연결 시 "-" 으로 연결
    print(result)


def list_comprehensions():
    """ List Comprehensions """
    print("\n===============[ List Comprehensions ]===============")

    print("\n >> 1-Dimensional-List\n")

    result = [i for i in range(10)]
    print(result)
    result = [i for i in range(10) if i % 2 == 0]  # if 뒤에는 조건으로, 조건에 만족하는 값을 list 값에 넣을 수 있음 (필터)
    print(result)

    print()
    word_1 = "Hello"
    word_2 = "World"
    result = [i + j for i in word_1 for j in word_2]  # Nested For loop
    """
    for i in word_1:
        for j in word_2:
            i + j
    와 같다.
    """
    print(result)

    print()
    case_1 = ["A", "B", "C"]
    case_2 = ["D", "E", "A"]
    result = [i + j for i in case_1 for j in case_2]
    print(result)
    result = [i + j for i in case_1 for j in case_2 if not (i == j)]  # i 와 j 가 같다면 list 에 추가하지 않음
    print(result)
    result.sort()
    print(result)

    print()
    words = "The quick brown fox jumps over the lazy dog".split()
    print(words)

    print("\n >> 2-Dimensional-List\n")

    # list 의 각 element 들을 대문자, 소문자, 길이 로 변환하여 2-Dimensional-List 로 변환
    stuff = [[w.upper(), w.lower(), len(w)] for w in words]
    print(stuff)

    print("\n >> Comparing two methods\n")
    case_1 = ["A", "B", "C"]
    case_2 = ["D", "E", "F"]
    print("case_1 :", case_1, "\ncase_2 :", case_2)

    result = [a + b for a in case_1 for b in case_2]
    print(result)
    result = [[a + b for a in case_1] for b in case_2]  # 즉, 뒤부터 기준이 되는 것을 알 수 있다.
    print(result)


def enumerate_zip():
    """ Enumerate """
    # list 의 element 를 추출할 때 번호를 붙여서 추출. >> 이 때 반환값은 tuple 의 형태임 !!!
    print("\n===============[ Enumerate ]===============")

    # list 에 있는 index 와 값을 unpacking
    for i, j in enumerate(["tic", "tac", "toe"]):
        print(i, j)

    # list 에 있는 index 와 값을 unpacking 하여 List 로 저장 (value 는 Tuple 이다.)
    mylist = ["a", "b", "c", "d"]
    mylist_enum = list(enumerate(mylist))
    print(mylist_enum)

    # 문장을 list 로 만들고, List 의 index와 값을 unpacking 하여 dict 로 저장
    str = "Hanyang University is an academic institute located in South Korea."
    words = str.split()
    words_enum = {i: j for i, j in enumerate(words)}
    print(words_enum)

    """ Zip """
    # 두 개의 list 의 값을 병렬적으로 추출함
    print("\n===============[ Zip ]===============")

    # 병렬적으로 값을 추출
    alist = ["a1", "a2", "a3"]
    blist = ["b1", "b2", "b3"]
    for a, b in zip(alist, blist):
        print(a, b)

    # 각 tuple 의 같은 index 끼리 묶음
    (a, b, c) = zip((1, 2, 3), (10, 20, 30), (100, 200, 300))
    print(a, b, c)

    zip_sum = [sum(x) for x in (a, b, c)]
    print(zip_sum)
    zip_sum = [sum(x) for x in zip((1, 2, 3), (10, 20, 30), (100, 200, 300))]
    print(zip_sum)

    """ Enumerate & Zip """
    # Enumerate 와 Zip 을 동시에 써서, 순서쌍을 나타낼 수 있음
    print("\n===============[ Enumerate & Zip ]===============")
    alist = ["a1", "a2", "a3"]
    blist = ["b1", "b2", "b3"]
    for i, (a, b) in enumerate(zip(alist, blist)):
        print(i, a, b)


def lambda_map_reduce():
    """ Lambda """
    # 함수 이름 없이, 함수처럼 쓸수 있는 익명함수
    print("\n===============[ Lambda ]===============")
    f = lambda x, y: x + y
    print(f(1, 4))
    f = lambda x: x ** 2
    print(f(3))
    f = lambda x: x / 2
    print(f(3))
    print((lambda x: x + 1)(5))

    """ Map """
    # Sequence 자료형의 각 element 에 동일한 function 을 적용함
    # 또한, Python3 에선 반드시 list 나 tuple 같은 sequence 자료형으로 반환을 해줘야 사용 가능하다.
    # (하지만 iterator 로 하나하나씩 호출 가능하다.)
    print("\n===============[ Map ]===============")

    ex = [1, 2, 3, 4, 5]
    print(ex)
    f = lambda x: x ** 2
    print(list(map(f, ex)))
    f = lambda x, y: x + y
    print(list(map(f, ex, ex)))

    # lambda 에도 필터를 넣을 수 있는데, 반드시 else 문을 넣어서 예외 케이스를 명시해줘야 한다.
    # (하지만 list comprehension 으로 표현할 수 있으므로 번거로운 상황에선 굳이 쓰지 않아도 된다.)
    print()
    f = lambda x: x ** 2 if x % 2 == 0 else x
    print(list(map(f, ex)))

    # Sequence 자료형으로 반환하는 것이 아니라, iterator 를 이용해서 하나하나씩 호출 가능하다.
    print()
    f = lambda x: x ** 2
    print(list(map(f, ex)))
    for i in map(f, ex):
        print(i)

    """ Reduce """
    # map 과 달리 sequence 자료형에 똑같은 함수를 적용해서 앞의 element 부터 누적하여 계산
        # from functools import reduce 를 해야 함
    print("\n===============[ Reduce ]===============")

    print(reduce(lambda x, y: x + y, [1, 2, 3, 4, 5]))

    def factorial(n):
        return reduce(lambda x, y: x * y, range(1, n + 1))

    print()
    print(list(range(1, 5)))
    print(factorial(5))


def asterisk():
    """ Asterisk """
    # '*' 를 의미한다. / 단순 곱셈, 제곱연산, 가변 인자 활용, unpacking 등 다양하게 사용된다.
    # *list :: list 를 unpacking 한다.
    # **dict :: dict 를 unpacking 한다.
    print("\n===============[ Asterisk ]===============")

    def asterisk_test1(a, *args):
        print(a, args)
        print(type(args))

    print("\n >> *args\n")
    print(asterisk_test1(1, 10, 11, 12, 13, 14))  # tuple 의 형태이다.
    """
    < 가변 인자 >
    1 은 a 에 할당되고,
    나머지는 tuple 형태로 *args 에 할당된다.
    """

    def asterisk_test2(a, **kwargs):
        print(a, kwargs)
        print(kwargs["b"], kwargs["c"])
        print(type(kwargs))

    print("\n >> **kargs\n")
    print(asterisk_test2(1, b=10, c=11, d=12, e=13, f=14))
    """
    < 키워드 인자 >
    1 은 a 에 할당되고,
    나머지는 dict 형태로 **kargs 에 할당된다.
    (이 때,
    1 은 변수명 a 에 할당되고,
    나머지 10, 11, 12, 13, 14 는 각각 변수명 b, c, d, e, f 에 할당된다.
    
    즉, **kargs 에 a=10 을 정의하려 하면 오류가 발생한다 !! << 중요.
    """

    # 또한 unpacking 을 좀 더 효율적으로 할 수 있다.
    def asterisk_test1_2(a, args):
        print(a, *args)
        print(type(args))

    print()
    asterisk_test1(1, (10, 11, 12, 13, 14))  # tuple 의 tuple 형태로 반환됨
    asterisk_test1(1, *(10, 11, 12, 13, 14))  # 미리 unpacking 된 형태로 입력시키기에, 일반 가변 인자를 입력하는 것과 같다.
    asterisk_test1_2(1, (10, 11, 12, 13, 14))  # tuple 형태로 들어가지만, 240 줄에서 unpacking 되어 반환된다.

    # list 를 직접 인자로 받거나, dict 를 직접 **kwargs 의 인자로 받을 수 있다.
    print()
    (a, b, c) = ([1, 2], [3, 4], [5, 6])
    print(a, b, c)

    print()
    data = ([1, 2], [3, 4], [5, 6])
    print(*data)

    print()
    print(asterisk_test1(1, *data))

    print()
    data = {"b": 20, "c": 21, "d": 22, "e": 23}
    print(asterisk_test2(1, **data))  # 이러면 b, c, d, e 가 각각 asterisk_test2 의 인자로 활용된다.

    # unpacking 을 통해 zip 으로 쉽게 활용 가능하다
    print()
    for data in zip(*([1, 2], [3, 4], [5, 6])):
        print(data)
        print(sum(data))
