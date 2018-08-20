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

"""

[ performance_measure_lec.ipynb ]


== (1) :: Performance Measure ==

- 만든 모델이 얼마나 실제 값을 잘 대변해주는가를 상대적으로 비교를 하기 위해 만든 지표

 여러가지 지표들이 존재
  - Mean Absolute Error
    :: 잔차의 절대값을 sum 하는 방식
       (0 에 가까울 수록 높은 적합도를 가짐)
  
  - Root Mean Squared Error (RMSE)
  >> 가장 많이 사용되는 방법
    :: 잔차 제곱의 sum 의 루트
       (0 에 가까울 수록 높은 적합도를 가짐)
  
  - R-Squared
    :: 기존 y 값과 예측 y^ 값과의 차이를 상관계수로 표현한 것
       (1 에 가까울 수록 높은 적합도를 가짐) 


== (2) :: Training & Test Data Set ==

- Training 한 데이터로 다시 Test 를 할 경우, Training 데이터에 과도하게 fitting 된 모델을 사용하게 될 가능성이 높다.
  >> 따라서 새로운 데이터가 출현했을 때, 기존 모델과의 차이가 존재함
- 모델은 새로운 데이터가 처리 가능하도록 generalize 되야 함
- 이를 위해 Training Set 과 Test Set 을 분리함

< 과정 >
 1. 데이터 set 을 Training set 과 Test set 으로 나눔
 2. Training set 을 가지고 모델을 만듬
 3. 완성된 모델을 RMSE 와 같은 지표로 평가한다.
 (weight 의 값에 따라 model 이 다양하게 생성되므로, Test set 으로 평가한다.)


== (3) :: Hold-out Method(Sampling) ==

>> Training data set 과 Test data set 을 나누는 과정

- 데이터를 Training 과 Test 로 나눠서 모델을 생성하고 테스트하는 기법
- 가장 일반적인 모델 생성을 위한 데이터 랜덤 샘플링 기법
- Training 과 Test 를 나누는 비율은 데이터의 크기에 따라 다름
- 일반적으로 Training data 2/3, Test data 1/3 을 활용함

"""

"""

[ sklearn_lec.ipynb ]

"""


def main():
    return None


if __name__ == "__main__":
    main()
