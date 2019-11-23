import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, CuDNNLSTM, Dropout

from nlu_seq2seq_load_data import load_atis
from nlu_seq2seq_vocab_size import get_vocab_size
from nlu_seq2seq_create_tensors import create_tensors
# from nlu_seq2seq_training import plot_training_accuracy


def run():
    # load data
    t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
        input_tensor_train, target_tensor_train, \
        query_data_train, intent_data_train, intent_data_label_train, slot_data_train \
        = load_atis('ms_cntk_atis.train.pkl')

    t2i_test, s2i_test, in2i_test, i2t_test, i2s_test, i2in_test, \
        input_tensor_test, target_tensor_test, \
        query_data_test, intent_data_test, intent_data_label_test, slot_data_test = load_atis('ms_cntk_atis.test.pkl')

    # vocab
    vocab_in_size, vocab_out_size = get_vocab_size(t2i_train, t2i_test, s2i_train, s2i_test)

    # tensors
    input_data_train, teacher_data_train, target_data_train, \
        len_input_train, len_target_train = create_tensors(input_tensor_train, target_tensor_train)
    input_data_test, teacher_data_test, target_data_test, \
        len_input_test, len_target_test = create_tensors(input_tensor_test, target_tensor_test, max_len=len_input_train)

    # build model
    # nlu_seq2seq_model.py
    BUFFER_SIZE = len(input_data_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024

    # encoder layers
    encoder_inputs = Input(shape=(len_input_train,))
    encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
    encoder_lstm = CuDNNLSTM(units=units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb(encoder_inputs))
    encoder_states = [state_h, state_c]

    # decoder layers
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
    decoder_lstm = CuDNNLSTM(units=units, return_sequences=True, return_state=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)

    # dense layers for inference
    decoder_d1 = Dense(units, activation='relu')
    decoder_d2 = Dense(vocab_out_size, activation='softmax')
    decoder_out = decoder_d2(Dropout(rate=.4)(decoder_d1(Dropout(rate=.4)(decoder_lstm_out))))

    model = Model([encoder_inputs, decoder_inputs], decoder_out)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.summary()

    # train model
    # nlu_seq2seq_training.py
    EPOCHS = 30
    history = model.fit([input_data_train, teacher_data_train], target_data_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=([input_data_test, teacher_data_test], target_data_test))
    # plot_training_accuracy(history)


if __name__ == '__main__':
    run()
