# https://towardsdatascience.com/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, CuDNNLSTM, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from nlu_seq2seq_load_data import load_atis
from nlu_seq2seq_vocab_size import get_vocab_size
from nlu_seq2seq_create_tensors import create_tensors
from nlu_seq2seq_training import plot_training_accuracy

from nltk.translate.bleu_score import corpus_bleu

def run():
    # load data
    t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
    input_tensor_train, target_tensor_train, \
    query_data_train, intent_data_train, intent_data_label_train, slot_data_train = load_atis('ms_cntk_atis.train.pkl', verbose=False)

    t2i_test, s2i_test, in2i_test, i2t_test, i2s_test, i2in_test, \
    input_tensor_test, target_tensor_test, \
    query_data_test, intent_data_test, intent_data_label_test, slot_data_test = load_atis('ms_cntk_atis.test.pkl', verbose=False)

    # vocab
    vocab_in_size, vocab_out_size = get_vocab_size(t2i_train, t2i_test, s2i_train, s2i_test)

    # tensors
    input_data_train, teacher_data_train, target_data_train, \
    len_input_train, len_target_train  = create_tensors(input_tensor_train, target_tensor_train)
    input_data_test, teacher_data_test, target_data_test, \
    len_input_test, len_target_test  = create_tensors(input_tensor_test, target_tensor_test, max_len=len_input_train)

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

    # build model & load weights
    model = Model([encoder_inputs, decoder_inputs], decoder_out)
    # model.load_weights('./model_dir/seq2seq.h5')
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    history = model.fit([input_data_train, teacher_data_train], target_data_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_data=([input_data_test, teacher_data_test], target_data_test))

    ###
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

    inf_decoder_inputs = Input(shape=(None,), name='inf_decoder_inputs')
    state_input_h = Input(shape=(units,), name='state_input_h')
    state_input_c = Input(shape=(units,), name='state_input_c')
    decoder_res, decoder_h, decoder_c = decoder_lstm(
        decoder_emb(inf_decoder_inputs),
        initial_state=[state_input_h, state_input_c])
    inf_decoder_out = decoder_d2(decoder_d1(decoder_res))
    inf_model = Model(inputs=[inf_decoder_inputs, state_input_h, state_input_c],
                      outputs=[inf_decoder_out, decoder_h, decoder_c])


    def preprocess_query(w):
        w = w.rstrip().strip().lower()
        w = 'BOS ' + w + ' EOS'
        return w


    def query_to_vector(query, len_input=len_input_train, t2i=t2i_train):
        pre = preprocess_query(query)
        vec = np.zeros(len_input)
        query_list = [t2i[s] for s in pre.split(' ')]
        for i, w in enumerate(query_list):
            vec[i] = w
        return vec


    def predict_slots(input_query, infenc_model, infmodel,
                      len_input=len_input_train,
                      t2i=t2i_train, s2i=s2i_train, i2s=i2s_train,
                      len_target=len_target_train,
                      attention=False):
        sent_len = len(input_query.split())
        sv = query_to_vector(input_query, len_input, t2i)
        sv = sv.reshape(1, len(sv))
        [emb_out, sh, sc] = infenc_model.predict(x=sv)

        i = 0
        start_vec = s2i['O']
        stop_vec = s2i['O']
        cur_vec = np.zeros((1, 1))
        cur_vec[0, 0] = start_vec
        cur_word = 'BOS'
        output_query = ''

        while cur_word != 'EOS' and i < (len_target - 1) and i < sent_len + 1:
            i += 1
            if cur_word != 'BOS':
                output_query = output_query + ' ' + cur_word
            x_in = [cur_vec, sh, sc]
            if attention:
                x_in += [emb_out]
            [nvec, sh, sc] = infmodel.predict(x=x_in)
            cur_vec[0, 0] = np.argmax(nvec[0, 0])
            cur_word = i2s[np.argmax(nvec[0, 0])]
        return output_query


    # inference on one sentence
    # input_query = 'what is the cheapest flight from boston to san francisco'
    # print(predict_slots(input_query, encoder_model, inf_model))

    # BiLingual Evaluation Understudy (BLEU) test
    def evaluate_slot_filling(queries, true_slots,
                              len_input=len_input_test,
                              t2i=t2i_test, s2i=s2i_test, i2s=i2s_test,
                              len_target=len_target_test):
        predicted_slots = []
        for q in queries:
            s = predict_slots(q, encoder_model, inf_model, len_input, t2i, s2i, i2s, len_target)
            predicted_slots.append(s)

        print('BLEU-1: %f' % corpus_bleu(true_slots, predicted_slots, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(true_slots, predicted_slots, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(true_slots, predicted_slots, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(true_slots, predicted_slots, weights=(0.25, 0.25, 0.25, 0.25)))

    evaluate_slot_filling(query_data_test, slot_data_test)

if __name__ == '__main__':
    run()
