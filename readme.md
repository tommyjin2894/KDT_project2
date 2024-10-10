# 혐호 표현 분류 모델 (seq2seq with attention)

### 0. 데이터 출처
[unsmile dataset - 스마일 게이트](https://github.com/smilegate-ai/korean_unsmile_dataset)


### 1. 파일 hierarchy

```text
checkpoint - 현 모델의 체크포인트 폴더
data_prep - 전처리 관련
    data_prep_with_w2v_clus.ipynb - 데이터 EDA 및 클러스터링 과정
    tokenizing_make_copus.ipynb - 토크나이징 과정
model - 모델 모듈
    _1_lstm_with_attention_model.py - 인코더 디코더의 lstm, attention 모델
    _2_encoder_simple_model.py - 인코더 to dense 의 모델

0_README.ipynb
1_result_graph.ipynb - 각종 그래프를 그려주는 코드
2_1_model_1_to_fit.ipynb - seq2seq 14 코퍼스 모델의 학습
2_2_model_2_to_fit.ipynb - seq2dense 11 라벨 모델의 학습
3_1_model_3_to_fit.ipynb - seq2seq 19 코퍼스 모델의 학습 및 다양한 실험
4_2_predict_for_service.ipynb - 예측 기능만 담긴 코드 - 추후 서비스시 이용
```
---
### 2가지 모델과, 3가지 방법

### 1. seq2dense 층의 모델
  - 사용 : ```2_2_model_2_to_fit.ipynb```
  - sigmoid 정답 라벨(중복 라벨이 있는 다중분류)
  
  |문장|종교|여성/가족|...|기타/혐오|남성|
  |---|---|---|---|---|---|
  |input 문장1|0|1|...|1|0|
  |input 문장2|1|1|...|1|0|
  ||||||
  |output|0.2|0.4|...|0.1|0.05|
        
### 2. seq2seq 층의 모델 (모델 input 과 output 모두 seq 데이터)
  - 방법 : 테이블 데이터의 Colums를 시퀀스로 변환 하여 문제 풀이
  - 사용 : ```2_1_model_1_to_fit.ipynb```
  - 정답 라벨의 corpus 14  
  - output : softmax 이용
    - 예시 \<start\> 인종/국적, 연령 \<end\>
  ```py
  {'padding': 0,'start': 1,'end': 2,'clean': 3,'종교': 4,'여성/가족': 5,'인종/국적': 6,...,'연령': 13}
  ```
### 3. seq2seq 층의 모델 (모델 input 과 output 모두 seq 데이터)
  - 방법 : 테이블 데이터의 Colums를 시퀀스로 변환하되, 붙어있던 라벨을 단어 단위롤 분리
  - 사용 : ```3_1_model_3_to_fit.ipynb```
  - 정답 라벨의 corpus 19
  - output : softmax 이용
    - 예시 \<start\> 인종, 국적, 연령 \<end\>
  ```py
  {'padding': 0,'start': 1,'end': 2,'clean': 3,'종교': 4,'여성': 5,'가족': 6,'인종': 7,...'연령': 18}
  ```
---
### 모델의 핵심 (seq2seq 모델이 분류모델에 적용 될 수 있었던 이유 - 예상)

<div style="text-align: center;">
    <img src="0_2_3 조 모델 구조 설명도_1.png" height="600">
</div>

- 인코더 부분 : 양방향 LSTM 의 최종 결과 값(모든 정방향, 역방향의 hs, cs 를 출력)
```py
    encoder_lstm = Bidirectional(LSTM(128, return_state=True, return_sequences=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_dropout)
```
- 디코더 부분 : 순방향 LSTM 의 최종 결과 값(모든 hs, cs 를 출력)
```py
    decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=encoder_states)
```
- 어텐션 레이어 : 모든 hs, cs 를 보고 정답을 맞추려면 어떤 값에 좀더 무게가 실려있어야 하는지 파악
```py
    attention_layer = Attention()([decoder_outputs, encoder_outputs])
```
---
### 각 모델별 성능 비교
- 모델 훈련시 정확도 및 test 데이터 에 대한 정확도
1. 심플 인코더 to dense (기본)모델
    - 훈련시 val_accuracy : 60.83 %
    - test 데이터에 대한 정확도: sigmoid: 트레쉬 홀드가 0.30일때 정확도 46 %

2. seq2seq 14 corpus
    - 훈련시 val_accuracy : 79.22 %
    - test 데이터에 대한 정확도 : 61.33 %
    
3. seq2seq 19 corpus
    - 훈련시 val_accuracy : 84.38 %
    - test 데이터에 대한 정확도  : 63.25 %

### 고찰
1. 각 모델에 대하여 훈련데이터, valid 데이터, 테스트 데이터의 정확도 편차가 크다. (단순한 모델인 1번 모델에서도 같은 현상이 나타났다.)

2. 특히 seq2seq 에서의 편차가 크다
    - 고찰 1 : 교사 강요로 학습할 때 이전 예상의 다음 예상이 아닌 정답에 대한 다음 seq 예상이므로 이에 따라 정확도의 큰 차이를 보인다.
    - 고찰 2 : 모델의 구조 및 predict 과정의 알고리즘 문제의 가능성이 있다.
3. 토크나이징의 문제
    - 좀더 좋은, 좀더 디테일한 분류가 가능한 토크나이징을 사용한다면 더 좋은 결과를 보여줄 것으로 예상 된다.
4. 추후 모델을 좀더 정제 하고, 층들을 추가하여 다양한 실험으로 더 나은 결과를 보여줄 것이라 예상
5. 참고 자료
    - [교사 강요에 대한 논문](https://arxiv.org/abs/1905.10617)
        - 요약 : 교사 강요로 인한 노출 편향(훈련 데이터와 테스트 데이터와의 편향성)에 대한 걱정은 크게 하지 않아도 된다.
            - 교사 강요뿐 아니라 다른 원인도 크게 작용 할 것 으로 예상.
### 데이터에 대한 문제점과 한계
1. 오분류 되어있는 데이터가 많이 있었다.
2. 묶여있는 (여성/가족), (악플/욕설) 등의 디테일한 세분화로 라벨링 되어 있지 않았다

- 그럼에도 불구하고, 인코더 디코더 어텐션 레이어가 문장의 의미를 잘 파악하여 성능은 나쁘지 않았다.
- 데이터를 조금더 정확하고 디테일하게 수집한다면 더욱 성능이 좋아 질 것이라고 판단

### 피드백

1. 멘토님 피드백
    1. 좋은 데이터, 모델 구성은 좋았다.
    2. **하지만 형태소 분석, 단어 빈도수, 코퍼스 정도밖에 하지 않아서 아쉬웠다.**
        - **너무 모델에 집중한 나머지, 입력데이터 전처리에 대한 처리를 많이 하지 못했다.**
2. 멘토님 피드백 2
    1. 데이터 수집 9페이지에 문장마다의 단어 개수를 표현 해 주셨는데 이것을 보고 padding을 위한 maxlen을 어떻게 설정해 두었는지 알려주었으면 좋았을 듯 하다. 
    2. 기본 Lstm 인코더 구조 부분(이전에 실험했던 내용) 부분도 내용에 첨가 되었으면 좋았을 듯 싶다. ⇒ 비교군으로
    3. 교사강요 방식으로 학습 진행 잘하셨고 그러면 나중에 테스트를 할 경우에는 어떻게 들어가는지 조금 더 상세하게. 설명 했으면 더 좋았을 듯 싶다.
    -> 전체적으로 굉장히 좋았다.
        - **좀더 디테일 한 설명을 내가 직접 했어도 좋았겠다.**
3. 강사님의 총평
    1. 결론 : 결과적으로 모델의 입력 값, 출력 값, 테스크 등 최종적으로 결과 이미지를 안 했다. 이부분을 넣어야 한다.
    2. 결과 예시: 
        
        ```python
        moonjang_ = '뭐 이런 게 다있어 이거 정신 나간거 아니야?'
        model_lstm_att.predict_from_seq(moonjang_, encoder_model, decoder_model, sets_for_predict)
        ```
        
        ['악플', '욕설']