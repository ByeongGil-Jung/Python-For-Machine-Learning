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

[ sklearn_GD_lec.ipynb ]

>> sklearn_GD_lec.ipynb 파일 참조

- sklearn 에는 다양한 Linear Regression 알고리즘이 내장되어 있다.

"""

"""

[ sklearn_SGD_lec.ipynb ]


== (1) :: Kinds of GD ==

** Stochastic Gradient Descent 마지막 부분에 4 가지 GD 방법에 대한 비교 및 차이가 잘 나와있다 !!

< Full-batch Gradient Descent >

- 한 점이 아닌, 한번에 여러 개의 점의 gradient 를 업데이트 하는 방법이다.

 (장점)
 - GD 가 1 개의 데이터를 미분하는 것과 달리, 동시에 여러 데이터를 미분한다.
 - 앞으로 일반적으로 GD = (full) batch GD 라고 가정한다.
 - 모든 데이터 셋으로 학습한다.
 - 업데이트 감소 -> 계산상 속도의 효율성을 높임
 - 안정적인 cost 함수 수렴
 
 (단점)
 - 지역 최적화 가능 (전체 data 를 같이 넣기 때문)
 - 메모리 문제 (-> 수십 억개의 데이터를 한번에 업데이트 하기엔 무리가 있음)
 - 대규모 data set (-> model / parameter 의 업데이트가 느려짐)
 
 >> 즉, 메모리 문제 등으로 딥러닝, 초대규모 data set 에선 사용하기 힘들다.
 >> 이를 보완할 방법으로 SGD 사용. (특히 mini-batch SGD 는 실제로 제일 많이 사용한다.)
 

< Stochastic Gradient Descent (SGD) >

- GD 처럼 차례대로 한 점씩 접근하는 것이 아닌, 한 번에 랜덤한 여러 점에서 접근한다.

 (장점)
 - 일부 문제에 대해 GD 보다 더 빨리 수렴
 - 지역 최적화 회피
 
 (단점)
 - 빈번한 업데이트로 인해 전체적으로 시간이 좀 오래 걸림
 - 대용량 데이터 작업 시 시간이 오래 걸림
 - 더 이상 cost 가 줄어들지 않는 시점의 발견이 어려움
 
 >> 이를 보완하기 위해 'Mini-batch (Stochastic) Gradient Descent' 개념이 나옴


< Mini-batch Stochastic Gradient Descent (mini-batch SGD) >

- SGD 의 단점을 보완하기 위해 나온 방식
- 처리할 데이터의 영역을 분할하여 연산을 수행한다.

- 한 번에 일정량의 데이터를 랜덤하게 뽑아서 학습
- SGD 와 Batch GD 를 혼합한 방법
- 가장 일반적으로 많이 쓰이는 방법
  (딥러닝의 optimization 된 알고리즘의 기본이다.)


== (2) :: Epoch & Batch-size ==

- Epoch ?
  :: 전체 데이터가 Training data 에 들어가는 횟수 (들어갈 때 카운팅)
- Full-batch 를 n 번 실행하면 n epoch
- Batch-size ?
  :: 한 번에 학습되는(update 되는) 데이터의 갯수
  
ex) 총 5,120 개의 Training data 에 512 batch-size 면 몇 번 학습을 해야 1 epoch 가 되는가 ?
ans) 10 번


"""

"""

[ sklearn_exercise.ipynb ]

>> sklearn_exercise.ipynb 파일 참조

- sklearn 라이브러리를 통해 다양한 Linear Regression 모델을 사용할 수 있다.

"""


def main():
    return None


if __name__ == "__main__":
    main()
