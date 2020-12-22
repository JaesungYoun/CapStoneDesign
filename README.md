# 머신러닝을 활용한 SMS 필터링

## 1. Introduction 

### 주제 선정 배경 

수많은 정보가 오가는 정보화시대에 불필요한 정보를 담은 SMS, 피싱 SMS 등으로 인하여 금전적인 피해 및 개인정보 유출 등의 피해를 보는 사람들이 늘어나고 있다. 한 통계 수치에 따르면 12,500,000통의 상품 판매 목적 스팸 SMS을 보내면, 한 건 정도는 매출로 이어진다고 한다. 따라서 이러한 피해를 입는 것을 방지하기위해 이메일을 필터링하여 스팸 SMS과 정상 SMS으로 분류할 필요가 있다.

### 주요 내용

1) Kaggle의 SMS 데이터셋을 활용하여 SMS 데이터를 수집
2) 공백, 숫자, 문장기호, 불용어(stop words), 단어 원형화, 소문자로 변환과 같은 데이터 전처리
3) 스팸, 정상 SMS 별로 자주 등장하는 단어들을 워드클라우드로 시각화
4) Naive Bayes, SVM(서포트 벡터 머신), MLP(다층 퍼셉트론) 모델들을 Naive Bayes 모델의 경우, K-fold Cross Validation(k-fold 교차검증)을 활용하여 parameter 변경해가며 정확도 측정, SVM모델의 경우 K-fold Cross Validation(k-fold 교차검증)을 활용하여 4가지 kernel로 정확도 측정, MLP 모델의 경우 solver parameter를 변경해가며 정확도 측정

### 최종 목표 
3가지 분류 모델에 데이터를 학습시켜 스팸 SMS와 정상 SMS를 분류하고 각 모델의 성능을 비교 및 분석하여 SMS 분류에 있어 가장 적합한 모델이 무엇인지 알아본다.
사용자의 SMS 활용 목적에 따라 적합한 모델이 무엇인지 분석해본다.

## 2. Data Analysis

### 데이터 수집, 전처리 및 시각화

1. 데이터를 불러온 후 분석하는데 있어 불필요한 컬럼을 제거한다.

    ![data](https://user-images.githubusercontent.com/73388615/102813074-87497780-440b-11eb-9a99-8392a6afe3b7.JPG)

2. 공백, 숫자, 문장기호, 불용어(stop words), 단어 원형화, 소문자로 변환한다
3. 스팸 SMS에 자주 등장하는 단어 워드 클라우드 시각화

    
    ![spamwords](https://user-images.githubusercontent.com/73388615/102811543-d80ba100-4408-11eb-9519-0daa240bf072.JPG)



### 모델 학습을 위한 데이터 준비 및 변형

1. Train Data와 Test Data 으로 분리 
    - Train Data로 학습을 하고 Test Data로 모델의 정확도 및 성능 측정하기 위함
2. CountVectorizer 를 활용하여 SMS 데이터를 Token Count 벡터로 변환 
    - 단어의 출현 빈도로 분류하기 메일을 분류하기 위함 
3. TfidfTransformer을 활용하여 Count Matrix를 tf-idf 표현으로 변환
    - TfidTransformer은 count matrix를 정규화 된 tf 또는 tf -idf 표현으로 변환시켜주는 사이킷런의 함수이다. SMS에서 매우 자주 발생하는 토큰의 영향을 축소시켜 해당 토큰들이 분류하는데 있어 큰 영향을 미치지 않도록 해야 한다. 적게 등장하는 단어일수록 해당 SMS가 정상인지 스팸인지 분류하는데 있어 유용하기 때문이다.
  
    - TF-IDF
      :  순서에 상관없이 Bag-of-word 형태의 Document-Term matrix에 형태에서 중요한 단어(변수)를 선택하는 방식을 TF-IDF(Term Frequency - Inverse Document Frequency)이라고 한다. 단어가 몇 번 등장 했는지에 대한 정보를 TF(frequency)로 정의 한다. 만약 어떤 단어가 언급된 문서의 수가 적다면 그 단어는 문서를 분류하는데 있어서 중요한 단어가 될 것이다. 따라서 그 문서 빈도의 역수 IDF(Inverse Document Frequency)를 TF의 곱으로 표현하여 등장횟수도 많고 문서 분별력 있는 단어들을 스코어링 한 것이다.
  
    ### 장점
      - 선택된 단어는 TF-IDF 스코어를 가지며 어떤 단어가 중요한 단어인지 직관적으로 해석이 가능하며, 전처리가 잘 수행 되었을 때 다른 변수선택/추출보다 견줄 만한 성능을 가지고 있다.
      - CountVectorizer의 단점을 보완


### 모델 학습 
1. Naive Bayes 모델 학습 
    - alpha(라플라스 스무딩), 사전확률 포함 여부를 변경해가면서 정확도 측정하였다.
    - 모델에 대한 과적합(Overfit), 데이터 편중을 막고자 K-Fold Cross Validation을 이용하였다.              
    ![K-CROSS](https://user-images.githubusercontent.com/73388615/102813728-b01e3c80-440c-11eb-911e-baa6682d0370.JPG)
    - Confusion Matrix를 활용하여 True Positive(TP), True Negative(TN), False Positive(FP), False Negative(FN)의 빈도를 알아보았다. 
  
2. SVM 모델 학습

    - Kernel(4가지) Linear,Poly,Sigmoid,Rbf로 측정하여서 어떤 분류선이 분류하는데 있어 가장 좋은지 알아보았다. 
    - 모델에 대한 과적합(Overfit), 데이터 편중을 막고자 K-Fold Cross Validation을 이용하였다. 
    - Confusion Matrix를 활용하여 True Positive(TP), True Negative(TN), False Positive(FP), False Negative(FN)의 빈도를 알아보았다. 
3. MLP 모델 학습

    ![MLP](https://user-images.githubusercontent.com/73388615/102813546-56b60d80-440c-11eb-8c2d-f8bfbe5dba51.JPG)

    - XOR 연산 같은 비선형적을 분리되는 데이터에 대해서 제대로 된 학습이 불가능하다. 이를 극복하기 위한 방안으로, 입력층과 출력층 사이에 은닉층을 두어 비선형적으로 분리되는 데이터에 대해서 학습이 가능하도록 하는 MLP(다층 퍼셉트론)를 사용.
    - solver parameter 를 변경해가면서 성능 및 정확도 측정하여 어떤 solver일때, 가장 정확도가 높은지 분석해보았다.
    - Confusion Matrix를 활용하여 True Positive(TP), True Negative(TN), False Positive(FP), False Negative(FN)의 빈도를 알아보았다. 


## 3. Results
   ### 결과 분석

   ![ROC_NaiveBayes](https://user-images.githubusercontent.com/73388615/102812069-c4ad0580-4409-11eb-9e89-93d1b0c90043.JPG)![ROC_SVM(Linear)](https://user-images.githubusercontent.com/73388615/102812072-c4ad0580-4409-11eb-98e6-fd066f2ec623.JPG)![ROC_MLP](https://user-images.githubusercontent.com/73388615/102812065-c37bd880-4409-11eb-9c7a-a0b8315db1dc.JPG)
        
      - 스팸 SMS에는 금전적인 단어가 많이 등장하고, SVM 모델의 Poly Kernel을 제외하고는 모든 모델들의 정확도는 97% 이상으로 높았다. 
        일반적으로 SVM 모델이 Naive Bayes 보다 3배정도 오랜 학습 시간을 필요로 하지만 예측은 더 빠르다. 
        따라서 어떤 모델이 분류에서 더 우수한지는 시나리오에 따라 다르다. 
        예를 들어, 신용카드 거래에서 정확도보다 신속하게 위조 결제에 대응하기 위해서는 예측이 빠른 SVM 모델이 더 좋을 것이다. 
        반대로 내가 진행하는 주제인 스팸 SMS를 분류하는 경우에는 예측이 빠른 것보단 정확도가 중요하므로 정확도가 더 높은 모델일수록 더 우수할 것이다. 
      - 미세하지만, 성능이 가장 좋은 모델 3가지는 MLP, SVM(Linear), Naive Bayes이다. 단, MLP, SVM 모델은 학습 시간이 오래 걸린다.
      - 업무에 관련한 메일을 놓치면 안되는 사람의 경우, FN(정상메일을 스팸메일로 예측한 경우)이 낮게 측정되는 모델이 적합하다.
      - 보이스 피싱과 같은 메일을 받고 싶지 않은 사람의 경우는 FP(스팸메일을 정상메일로 예측한 경우) 가 낮게 측정되는 모델이 적합하다.
      - 더욱 성능이 좋은 분류를 하려면 FN,FP를 동시에 최대한 낮추는 것이 중요하다.
      



## 4. Expected effect and Application plan
### Expected effect
학습시킨 모델의 스팸 SMS 분류를 통한 피해 발생을 방지할 수 있다.
### Application plan    
이러한 모델들을 활용하여 Kaggle 데이터 셋에 국한된 SMS 뿐만 아니라, 실제 현실에서 수신하는 SMS 및 이메일, 메신저들도 분류를 할 수 있도록한다.
        
## 5. Conclusion

이번 데이터캡스톤디자인 프로젝트로 머신러닝 분류 모델 3가지를 활용하여 SMS 분류를 하고 성능 비교 및 분석을 하였는데 모두 성능이 우수하게 나와서 SVM(Poly Kernel)을 제외한 3가지 모델 모두 SMS를 분류하는 데 있어 우수한 모델인 것을 알게 되었다. 아쉬웠던 점은 kaggle 데이터 셋의 크기가 작을뿐더러 이 데이터 셋에만 국한된 SMS 밖에 분류를 하지 못하였고, False Negative, 
False Positive를 동시에 낮추는 방법을 찾지 못하였다. 또한, 처음으로 머신러닝 분야를 공부해보기 때문에 이미 검증된 모델들로 비교적 쉽게 분석을 진행하여서 좀 더 유의미하고 창의적인 분석을 하지 못한 데에 아쉬움이 많았다. 그래도 이번 캡스톤디자인 프로젝트를 진행하기 전에는 데이터 분석을 제대로 해본 경험이 한번도 없어서 정말 낯설고 어떻게 데이터 분석을 해야하고 어떻게 스터디를 해야할 지 조차 감이 잘 안잡혔다면, 캡스톤디자인을 진행하고 나서는 데이터 분석과 스터디 방향에 대하여 많은 부분을 깨달아서 방학동안 데이터 분석 스터디를 열심히 하여 다음에는 더 의미 있고 심도 있는 캡스톤디자인을 진행하려고 한다.










