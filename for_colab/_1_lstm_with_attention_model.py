from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Bidirectional, Attention, Concatenate, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from konlpy.tag import Okt


# test 1 
def seq2seq_with_attention(max_text_len, corpus_size, max_summary_len, corpus_size_answer, **kwargs):
    # 하이퍼파라미터 설정
    embedding_size1 = kwargs.get('embedding_size1', 1024)
    embedding_size2 = kwargs.get('embedding_size2', 1024)
    lstm_size = kwargs.get('lstm_size', 32)
    dropout_ratio = kwargs.get('dropout_ratio', 0.6)

    # 인코더
    encoder_inputs = Input(shape=(max_text_len,))
    encoder_embedding = Embedding(corpus_size + 1, embedding_size1, trainable=True, mask_zero=True)(encoder_inputs) # 패딩 마스킹
    encoder_dropout = Dropout(dropout_ratio)(encoder_embedding)
    encoder_lstm = Bidirectional(LSTM(lstm_size, return_state=True, return_sequences=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_dropout)
        #!!! 여기서 encoder_outputs는 모든 타임스텝의 히든 스테이트의 출력임 !!!

    encoder_h = Concatenate()([forward_h, backward_h])
    encoder_c = Concatenate()([forward_c, backward_c])
    encoder_states = [encoder_h, encoder_c]
    encoder_outputs = Dropout(dropout_ratio)(encoder_outputs)
    
    # 디코더
    decoder_inputs = Input(shape=(max_summary_len,))
    decoder_embedding_layer = Embedding(corpus_size_answer, embedding_size2, trainable=True, mask_zero=True) # 패딩 마스킹
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_dropout = Dropout(dropout_ratio)(decoder_embedding)
    decoder_lstm = LSTM(lstm_size*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state=encoder_states)
    decoder_outputs = Dropout(dropout_ratio)(decoder_outputs)
        #!!! 여기서 decoder_outputs도 마찬가지로 모든 타임스텝의 히든 스테이트의 출력임 !!!
    
    # 어텐션 메커니즘
        #!!! 인코더와 디코더의 각 타임 스텝의 히든 스테이트를 보고 !!!
        #어디에 집중(가중)할 지 계산?
    attention_layer = Attention()([decoder_outputs, encoder_outputs])
    
        #디코더 출력(히든 스테이트) 와 어텐션 결과(어느 히든 스테이트가 중요한지(가중치)) 의 콘켓
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_layer])
    
    # 출력층
    decoder_dense = Dense(corpus_size_answer, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input) # 최종 출력 출력과의 완전연결(디코더의 히든 스테이트와 + 어텐션 가중 값)
    
    # 모델 정의
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    #--------------------------------------------------------------------------
    #예측 및 요약을 위한 디코더 모델 설정
    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

    decoder_state_input_h = Input(shape=(lstm_size*2,))
    decoder_state_input_c = Input(shape=(lstm_size*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # 한 스텝씩 예측하기 때문에 decoder_inputs의 shape을 (1, 1)로 변경
    decoder_single_input = Input(shape=(1,))
    decoder_single_embedding = decoder_embedding_layer(decoder_single_input)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_single_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]


    # 어텐션
    attention_layer = Attention()([decoder_outputs, encoder_outputs])
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_layer])

    decoder_outputs = decoder_dense(decoder_concat_input)
    decoder_states = [state_h, state_c]

    decoder_model = Model(
        [decoder_single_input, encoder_outputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder_model, decoder_model

def predict_from_seq(moonjang_, encoder_model, decoder_model, sets_for_predict):
    sentences_corpus_word_index, max_text_len, label_corpus_word_index, label_corpus_index_word, max_label_len = sets_for_predict
    tokenizer = Okt()
    tokened_for_predict = tokenizer.morphs(moonjang_)
    tokened_for_predict = [sentences_corpus_word_index.get(token, 0) for token in tokened_for_predict]
    new_text_padded = pad_sequences([tokened_for_predict], maxlen=max_text_len, padding='post')
    
    # 인코더 모델을 사용하여 상태 벡터를 예측
    states_value = encoder_model.predict(new_text_padded, verbose=0)
    
    # 디코더의 초기 입력 설정 (시작 토큰 인덱스 사용)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = label_corpus_word_index.get("start", 0)
    
    # 예측을 위한 초기화
    stop_condition = False
    decoded_label = []
    init_out, init_h, init_c = states_value
    
    # 예측 반복
    while not stop_condition:
        # LSTM에 임베딩 벡터와 초기 상태를 입력으로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + [init_out, init_h, init_c], verbose=0)
        
        # 가장 가능성이 높은 토큰을 샘플링합니다.
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = label_corpus_index_word.get(sampled_token_index, '')
        
        # 종료 조건: 최대 길이에 도달하거나 종료 토큰을 샘플링한 경우
        if sampled_char == 'end' or len(decoded_label) >= max_label_len:
            stop_condition = True
        elif sampled_char != 'start':
            decoded_label.append(sampled_char)
        
        # 샘플링된 토큰을 다음 입력으로 사용합니다.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # 상태를 업데이트합니다.
        init_h, init_c = h, c
    
    return decoded_label
