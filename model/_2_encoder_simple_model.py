from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Bidirectional, Concatenate, Dense, BatchNormalization

def encoder_simple_model(input_shape_, corpus_size, output_shape_):
    # 인코더 와 마지막의 덴스층
    encoder_inputs = Input(shape=(input_shape_,))
    encoder_embedding = Embedding(corpus_size + 1, 32, trainable=True, mask_zero=True)(encoder_inputs) # 패딩 마스킹
    encoder_dropout = Dropout(0.3)(encoder_embedding)

    encoder_lstm = Bidirectional(LSTM(32, return_state=True, return_sequences=True))
    _, forward_h, _, backward_h, _ = encoder_lstm(encoder_dropout)
    encoder_h = Concatenate()([forward_h, backward_h])
    encoder_h = Dropout(0.3)(encoder_h)
    encoder_h = BatchNormalization()(encoder_h)

    layer_1 = Dense(128, activation='relu')(encoder_h)
    layer_1 = Dropout(0.3)(layer_1)
    layer_1 = BatchNormalization()(layer_1)

    decoder_output = Dense(output_shape_, activation='sigmoid')(encoder_h)

    # 모델 정의
    model = Model(encoder_inputs, decoder_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model