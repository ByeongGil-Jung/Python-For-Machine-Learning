"""
 Created by IntelliJ IDEA.
 Project: 4_Linear_Regression
 ===========================================
 User: ByeongGil Jung
 Date: 2018-07-23
 Time: 오후 11:13
"""

"""

< Gradient Descent >
- 기울기를 계속 측정하여 0 으로 수렴할 때까지 반복 계산하는 방법

"""

"""

[ gradient_descent_lec.ipynb ]

- a(알파) 는 learning rate 로, 한 번 계산하고 얼마만큼 더 내려가서 반복 계산할 것인지에 대한 수치

>> learning rate 에 대한 선정 (> 얼마나 많이 loop 을 돌 것인가 ?)
- 여러 극솟값이 있을 경우, 처음 초기 x 값의 위치를 잘 조정해서 넣어야 한다.
(잘 못 지정하면 수렴하지 않거나, 적절하지 않은 극솟값에 도달할 수 있다.)


== (1) :: Linear Regression with Gradient Descent ==

- 목적은 cost function (J) 를 최소화 시켜줘야 하는 점에서 같다.

- 임의의 t1, t2 값으로 초기화 (theta)
- cost function J(t1, t2) 가 최소화 될 때까지 학습
- 더 이상 cost function 이 줄어들지 않거나, 학습 횟수를 초과할 때 종료

- x_new 가 theta (w_j) 라고 생각하면 된다.
- 중요한 것은, x_new 와 x_old 가 동시적으로 업데이트 된다는 것이다. ( 중요 !! )
(old 값끼리 계산하고, new 값은 new 값들끼리 업데이트 된다는 의미.)

- learning rate, iteration 횟수 등, parameter 지정
- feature 가 많으면 normal equation 에 비해 상대적으로 빠르다. (사실은 체감 안된다.)
- 최적값에 수렴하지 않을 수 있다. (그러나 대부분 수렴한다.)


== (2) :: Multivariate Linear Regression ==

- 두 개 이상의 feature 로 구성된 데이터를 분석할 때
(-> 식은 많아지지만, 해야할 일은 cost 함수의 최적화로 동일하다.)

"""


def main():
    return None


if __name__ == "__main__":
    main()
